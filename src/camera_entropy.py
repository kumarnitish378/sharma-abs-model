"""Physical random numbers from a camera, usable as a drop-in RNG.

Pipeline:  camera frame -> LSB of every unsaturated pixel (sensor noise)
           -> SHA-256 conditioning, 128 raw bytes -> 32 output bytes, chained
           -> byte pool filled by a background thread

Use it in place of a software RNG:

    from camera_entropy import CameraEntropy, CameraRandom, CameraNumpyRNG
    ent = CameraEntropy(cam=0)            # or CameraEntropy(fake=True) without a camera
    rng = CameraRandom(ent)               # behaves like random.Random: randint, gauss, ...
    nrng = CameraNumpyRNG(ent)            # integers(), normal(), random() like numpy's Generator

Learning / simulation tool, not a certified TRNG (no NIST SP 800-90B health tests).
"""
import hashlib
import math
import random
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------- frame sources

class CameraSource:
    def __init__(self, index=0):
        if cv2 is None:
            raise RuntimeError("OpenCV not installed: pip install opencv-python")
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index} (try another index or fake=True)")
        self.name = f"camera {index}"

    def read(self):
        ok, frame = self.cap.read()
        return frame[:, :, ::-1] if ok else None   # BGR -> RGB

    def close(self):
        self.cap.release()


class FakeSource:
    """Simulated camera for testing without hardware. Its noise is PSEUDO-random."""

    def __init__(self, w=640, h=480):
        self.rng = np.random.default_rng()
        y, x = np.mgrid[0:h, 0:w]
        scene = np.stack([x / w * 200 + 20, y / h * 180 + 30, (x + y) / (w + h) * 160 + 40], -1)
        scene[60:180, 420:600] = 255          # saturated window (no entropy)
        scene[300:420, 60:260] = 90           # flat dark area
        self.scene = scene
        self.name = "FAKE simulated camera"

    def read(self):
        noisy = self.scene + self.rng.normal(0, 2.0, self.scene.shape)
        time.sleep(1 / 30)
        return np.clip(np.round(noisy), 0, 255).astype(np.uint8)

    def close(self):
        pass


# ---------------------------------------------------------------- conditioning

class Extractor:
    CHUNK = 128  # raw bytes per SHA-256 call -> 32 output bytes (4:1)

    def __init__(self):
        self.prev = b"\x00" * 32

    def process(self, frame):
        """Returns (raw LSB bytes, conditioned bytes, fraction of usable pixels)."""
        valid = (frame > 0) & (frame < 255)
        raw = np.packbits((frame[valid] & 1).astype(np.uint8)).tobytes()
        out = bytearray()
        h = self.prev
        for i in range(0, len(raw) - self.CHUNK + 1, self.CHUNK):
            h = hashlib.sha256(h + raw[i:i + self.CHUNK]).digest()
            out += h
        self.prev = h
        return np.frombuffer(raw, np.uint8), np.frombuffer(bytes(out), np.uint8), float(valid.mean())


# ---------------------------------------------------------------- entropy pool

class CameraEntropy:
    """Background thread that keeps a pool of conditioned camera bytes."""

    def __init__(self, cam=0, fake=False, pool_size=4 << 20, timeout=5.0):
        self.source = FakeSource() if fake else CameraSource(cam)
        self.name = self.source.name
        self.ex = Extractor()
        self.pool = bytearray()
        self.pool_size, self.timeout = pool_size, timeout
        self.lock = threading.Condition()
        self.produced = self.consumed = 0
        self.last_frame = None
        self.usable = 0.0
        self.running = True
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while self.running:
                frame = self.source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                _, out, usable = self.ex.process(frame)
                with self.lock:
                    self.last_frame, self.usable = frame, usable
                    room = self.pool_size - len(self.pool)
                    if room > 0:
                        self.pool += out.tobytes()[:room]
                        self.produced += min(room, len(out))
                    self.lock.notify_all()
        except Exception as e:  # surface camera errors to the consumer
            self.error = e
            with self.lock:
                self.lock.notify_all()

    def read(self, n):
        """Return exactly n conditioned bytes, waiting for the camera if needed."""
        out = bytearray()
        deadline = time.time() + self.timeout
        with self.lock:
            while len(out) < n:
                if self.pool:
                    take = min(n - len(out), len(self.pool))
                    out += self.pool[:take]
                    del self.pool[:take]
                    continue
                if self.error:
                    raise RuntimeError(f"camera entropy failed: {self.error}")
                if not self.lock.wait(max(0.0, deadline - time.time())) and not self.pool:
                    raise TimeoutError("no camera entropy: is the camera delivering frames?")
            self.consumed += n
        return bytes(out)

    def stats(self):
        with self.lock:
            return dict(produced=self.produced, consumed=self.consumed,
                        pool=len(self.pool), usable=self.usable)

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.source.close()


# ---------------------------------------------------------------- RNG adapters

class CameraRandom(random.Random):
    """random.Random whose every bit comes from the camera.

    randint, randrange, choice, shuffle, uniform, gauss ... all work unchanged.
    seed() is ignored: a physical source cannot be replayed.
    """

    def __init__(self, entropy):
        self.entropy = entropy
        super().__init__()

    def seed(self, *args, **kwargs):
        pass

    def random(self):
        return (int.from_bytes(self.entropy.read(7), "big") >> 3) * (1.0 / (1 << 53))

    def getrandbits(self, k):
        if k <= 0:
            return 0
        nbytes = (k + 7) // 8
        return int.from_bytes(self.entropy.read(nbytes), "big") >> (nbytes * 8 - k)

    def getstate(self):
        raise NotImplementedError("a physical entropy source has no replayable state")

    def setstate(self, state):
        raise NotImplementedError("a physical entropy source has no replayable state")


class CameraNumpyRNG:
    """The parts of numpy.random.Generator the simulations use, fed by the camera."""

    def __init__(self, entropy):
        self.entropy = entropy

    def _u32(self, count):
        return np.frombuffer(self.entropy.read(4 * count), dtype=">u4").astype(np.uint64)

    def integers(self, low, high=None, size=None):
        """Uniform integers in [low, high), unbiased (rejection sampling)."""
        if high is None:
            low, high = 0, low
        span = int(high) - int(low)
        if not 0 < span <= 1 << 32:
            raise ValueError("span must be in 1 .. 2**32")
        count = int(np.prod(size)) if size is not None else 1
        limit = ((1 << 32) // span) * span
        got = np.empty(0, dtype=np.uint64)
        while got.size < count:
            v = self._u32(max(16, int((count - got.size) * 1.1) + 1))
            got = np.concatenate([got, v[v < limit]])
        res = (got[:count] % span).astype(np.int64) + int(low)
        return res.reshape(size) if size is not None else int(res[0])

    def random(self, size=None):
        count = int(np.prod(size)) if size is not None else 1
        b = np.frombuffer(self.entropy.read(8 * count), dtype=">u8") >> np.uint64(11)
        res = b.astype(np.float64) * (1.0 / (1 << 53))
        return res.reshape(size) if size is not None else float(res[0])

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Box-Muller transform on camera uniforms."""
        count = int(np.prod(size)) if size is not None else 1
        m = (count + 1) // 2
        u1 = 1.0 - self.random(m)          # in (0, 1]
        u2 = self.random(m)
        r = np.sqrt(-2.0 * np.log(u1))
        z = np.concatenate([r * np.cos(2 * math.pi * u2), r * np.sin(2 * math.pi * u2)])[:count]
        res = loc + scale * z
        return res.reshape(size) if size is not None else float(res[0])


# ---------------------------------------------------------------- CLI helper for demos

def add_rng_args(parser):
    """Add --camera / --cam / --fake-camera options to an argparse parser."""
    g = parser.add_argument_group("random source")
    g.add_argument("--camera", action="store_true", help="use camera noise as the random source")
    g.add_argument("--cam", type=int, default=0, help="camera index (with --camera)")
    g.add_argument("--fake-camera", action="store_true",
                   help="use the simulated camera (pseudo-random; for testing without hardware)")
    return parser


def make_entropy(args):
    """CameraEntropy from parsed args, or None for the normal software RNG."""
    if args.camera or args.fake_camera:
        return CameraEntropy(cam=args.cam, fake=args.fake_camera)
    return None


def source_label(entropy):
    if entropy is None:
        return "software PRNG (Mersenne Twister)"
    s = entropy.stats()
    return f"{entropy.name}: {s['consumed'] / 1024:,.0f} KB used"
