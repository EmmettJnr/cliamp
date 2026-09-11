package player

import (
	"math"
	"sync/atomic"
	"time"

	"github.com/gopxl/beep/v2"
)

// fadeFloorDB is the attenuation at which the transport envelope is treated as
// silence.
const fadeFloorDB = -60

// fadeFloorLn converts fadeFloorDB into the natural-log domain so the envelope
// costs one Exp instead of a Pow.
var fadeFloorLn = fadeFloorDB * math.Ln10 / 20

// fadeEnvelope maps envelope position to linear gain, linear in dB (constant
// dB per second) rather than equal-power.
//
// Loudness perception is logarithmic, so a dB-linear ramp is the one that
// sounds evenly paced. An equal-power curve is the right choice for a
// crossfade, where two streams have to sum to constant power, but against
// silence it holds within a dB of unity for the first third of the ramp, which
// reads as the fade not starting for a beat.
func fadeEnvelope(pos float64) float64 {
	if pos <= 0 {
		return 0
	}
	if pos >= 1 {
		return 1
	}
	return math.Exp(fadeFloorLn * (1 - pos))
}

// fadeState is a transport fade envelope shared by the UI thread and the audio
// callback. pos walks toward target by step per sample and is owned by the
// audio thread, which publishes it once per Stream() call rather than per
// sample.
type fadeState struct {
	pos    atomic.Uint64 // float64 bits, 0..1
	target atomic.Uint64 // float64 bits, 0 (silent) or 1 (full)
	step   atomic.Uint64 // float64 bits, per-sample delta toward target
}

func newFadeState() *fadeState {
	f := &fadeState{}
	f.set(1)
	return f
}

// set jumps the envelope to pos with no ramp. A nil fadeState is inert, so a
// Player built without New (as tests do) simply has no fades.
func (f *fadeState) set(pos float64) {
	if f == nil {
		return
	}
	bits := math.Float64bits(pos)
	f.pos.Store(bits)
	f.target.Store(bits)
}

// rampTo starts a ramp toward target that takes d at the given sample rate.
func (f *fadeState) rampTo(target float64, d time.Duration, sampleRate int) {
	if f == nil {
		return
	}
	steps := float64(sampleRate) * d.Seconds()
	if steps < 1 {
		steps = 1
	}
	f.step.Store(math.Float64bits(1 / steps))
	f.target.Store(math.Float64bits(target))
}

// silent reports whether the envelope has fully faded out.
func (f *fadeState) silent() bool {
	if f == nil {
		return true
	}
	return math.Float64frombits(f.pos.Load()) == 0
}

// volumeStreamer applies dB gain and optional mono downmix to an audio stream.
// Volume and mono are read via atomic operations, eliminating mutex contention
// with the UI thread on the audio hot path.
type volumeStreamer struct {
	s          beep.Streamer
	vol        *atomic.Uint64 // dB stored as Float64bits
	mono       *atomic.Bool
	fade       *fadeState // transport fade envelope; nil disables fading
	cachedDB   float64    // last dB value used to compute cachedGain; starts NaN to force first compute
	cachedGain float64    // precomputed linear gain = 10^(dB/20)
}

func (v *volumeStreamer) Stream(samples [][2]float64) (int, bool) {
	n, ok := v.s.Stream(samples)
	if n == 0 {
		return 0, ok
	}
	db := math.Float64frombits(v.vol.Load())
	mono := v.mono.Load()
	// Recompute gain only when volume changes (rare) instead of every Stream() call.
	if db != v.cachedDB {
		v.cachedGain = math.Pow(10, db/20)
		v.cachedDB = db
	}
	gain := v.cachedGain

	// Fade envelope. Idle at full volume is the common case and costs one
	// comparison; fadeEnvelope runs only while a ramp is in flight or the
	// envelope is holding below unity.
	pos, target, step := 1.0, 1.0, 0.0
	fading := false
	if v.fade != nil {
		pos = math.Float64frombits(v.fade.pos.Load())
		target = math.Float64frombits(v.fade.target.Load())
		step = math.Float64frombits(v.fade.step.Load())
		fading = pos != target || pos != 1
	}

	for i := range n {
		g := gain
		if fading {
			if pos < target {
				pos = min(pos+step, target)
			} else if pos > target {
				pos = max(pos-step, target)
			}
			g *= fadeEnvelope(pos)
		}
		samples[i][0] *= g
		samples[i][1] *= g
		if mono {
			mid := (samples[i][0] + samples[i][1]) / 2
			samples[i][0] = mid
			samples[i][1] = mid
		}
	}
	if fading {
		v.fade.pos.Store(math.Float64bits(pos))
	}
	return n, ok
}

func (v *volumeStreamer) Err() error { return v.s.Err() }
