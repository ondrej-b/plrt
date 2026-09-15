#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plrt-psychopy.py -- Pupillary Light Reflex Task (PLRT)
======================================================
taVNS project, September 2026

PsychoPy (coder-style) implementation of the PLR task with direct EyeLink
integration through SR Research's ``pylink``.  Runs without a tracker in
"dummy mode" (leave the tracker address empty in the start dialog) and even
without ``pylink`` installed (a no-op stub is used, with a warning).

Design (all durations configurable in PARAMS below)
---------------------------------------------------
Session:
    dialog -> connect tracker, open EDF -> camera setup / calibration (once)
    -> instructions -> start recording (continuous for the whole session)
    -> 4 blocks -> stop recording, receive EDF, close.

Block (repeated N_BLOCKS times, stimulus order fixed 1 -> 4):
    PAUSE       black screen, no cross                 120 s
    BASELINE    black screen + grey fixation cross     U(20, 30) s
    STIM 1      white disc, diameter 1/8 screen height 200 ms   (cross off)
    ISI         black screen + grey fixation cross     U(15, 22) s
    STIM 2      white disc, diameter 1/4 screen height 200 ms
    ISI                                                U(15, 22) s
    STIM 3      white disc, diameter 1/2 screen height 200 ms
    ISI                                                U(15, 22) s
    STIM 4      full white screen                      200 ms
    ISI                                                U(15, 22) s

Stimulus durations are frame-counted (12 frames at 60 Hz); the long
intervals are clock-based but keep flipping every frame so that the
keyboard (Esc = abort) stays responsive.

EDF messages (timestamped by the Host PC on receipt, sent right after the
flip that made the event visible):
    SESSION_START <participant> <session>
    BLOCK_START <k>            BLOCK_END <k>
    PAUSE_ON <k>               PAUSE_OFF <k>
    CROSS_ON <k> <phase>       CROSS_OFF <k> <phase>      phase = baseline|isi
    TRIALID <k> <t>            (t = 1..4 within block)
    STIM_ON <k> <t> <name> <size_frac>
    STIM_OFF <k> <t> <name>
    SESSION_END | SESSION_ABORT
Every visible change of the display is thus bracketed by an ON/OFF pair.

Outputs (folder ``data/`` next to this script):
    <participant>_<session>_<date>_plrt.csv   event log from PsychoPy clocks
    <EDFNAME>.EDF                              pulled from the Host PC at the end

Run from a real terminal or the PsychoPy Runner, not from an inline IPython
console (see project notes on Spyder).
"""

import csv
import os
import random
import re
import sys
import time
from datetime import datetime

from psychopy import core, event, gui, visual

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
PARAMS = dict(
    n_blocks=4,
    pause_s=120.0,                 # black screen before every block (incl. block 1) 120
    baseline_range_s=(20.0, 30.0), # cross only, start of block
    stim_s=0.200,                  # each stimulus
    isi_range_s=(15.0, 22.0),      # cross only, after every stimulus
    # stimuli in presentation order: (name, diameter as fraction of screen height;
    # None = full white screen)
    stimuli=[
        ('circle_1_8', 1 / 8),
        ('circle_1_4', 1 / 4),
        ('circle_1_2', 1 / 2),
        ('fullscreen', None),
    ],
    bg_color=(-1, -1, -1),         # black
    stim_color=(1, 1, 1),          # white
    cross_color=(0.0, 0.0, 0.0),   # mid grey in PsychoPy's -1..1 scale
    cross_size_frac=0.03,          # arm length as fraction of screen height
    cross_line_px=3,
    screen=0,                      # monitor index for the stimulus window
    fullscreen=True,
    expected_hz=60.0,              # used if measurement is off or unreliable
    measure_refresh=True,          # False -> trust expected_hz, skip the ~2 s test
    tracker_ip='100.1.1.1',        # default in the dialog; '' -> dummy mode
    dummy_use_pylink=False,        # dummy mode: True exercises pylink's own dummy
                                   # connection (slow) incl. the calibration screen;
                                   # False bypasses pylink entirely (instant start)
    sample_rate=500,
    calibration_type='HV5',
    seed=None,                     # None -> random seed derived from time (logged)
    debug_speed=20.0,               # >1 shortens pause/baseline/ISI for testing (stim unchanged)
    abort_key='escape',
    confirm_keys=['space'],        # keyboard keys that confirm a screen ...
    use_cedrus=True,               # ... plus ANY button on a Cedrus response box (pyxid2)
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# --------------------------------------------------------------------------
# pylink (optional)
# --------------------------------------------------------------------------
try:
    import pylink
    HAVE_PYLINK = True
except ImportError:
    pylink = None
    HAVE_PYLINK = False


class DummyTracker(object):
    """No-op stand-in used when pylink is not installed at all."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return 0
        return _noop

    def isConnected(self):
        return False


class EyeLinkSession(object):
    """Thin wrapper around pylink for this task."""

    def __init__(self, ip, edf_name, win, params, log):
        self.win = win
        self.params = params
        self.log = log
        self.edf_name = edf_name
        self.recording = False
        self.dummy = (ip == '' or ip is None)
        self.genv = None

        if not HAVE_PYLINK:
            print('WARNING: pylink not available -> tracker calls are no-ops.')
            self.tracker = DummyTracker()
            self.dummy = True
            return

        if self.dummy and not params['dummy_use_pylink']:
            # Fast dummy mode: never touch pylink at all.  pylink's own dummy
            # connection still runs every openDataFile/sendCommand/doTrackerSetup
            # call against a non-existent Host PC, and each one blocks until it
            # times out -- that is what made startup take minutes.
            print('EyeLink: dummy mode (no tracker, pylink bypassed).')
            self.tracker = DummyTracker()
            return

        if self.dummy:
            print('EyeLink: dummy mode via pylink (slow: every command waits for '
                  'a timeout).')
            self.tracker = pylink.EyeLink(None)
        else:
            print('EyeLink: connecting to %s ...' % ip)
            self.tracker = pylink.EyeLink(ip)

        # --- open EDF on the Host PC ------------------------------------
        self.tracker.openDataFile(edf_name)
        self.tracker.sendCommand('add_file_preamble_text "PLRT taVNS project"')

        # --- tracker configuration --------------------------------------
        self.tracker.setOfflineMode()
        w, h = win.size
        self.tracker.sendCommand('screen_pixel_coords = 0 0 %d %d' % (w - 1, h - 1))
        self.tracker.sendMessage('DISPLAY_COORDS 0 0 %d %d' % (w - 1, h - 1))
        self.tracker.sendCommand('sample_rate %d' % params['sample_rate'])
        self.tracker.sendCommand('pupil_size_diameter = YES')   # diameter, not area
        self.tracker.sendCommand('calibration_type = %s' % params['calibration_type'])
        self.tracker.sendCommand(
            'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT')
        self.tracker.sendCommand(
            'file_sample_data = LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT')
        self.tracker.sendCommand(
            'link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,INPUT')
        self.tracker.sendCommand(
            'link_sample_data = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT')

        # --- calibration graphics inside the PsychoPy window ------------
        # Developers Kit <= 2.1: EyeLinkCoreGraphicsPsychoPy.py copied next to the
        # script.  Kit 2.2+: shipped as the pip package
        # `psychopy-eyelink-coregraphics` (part of the psychopy-eyelink plugin).
        CoreGraphics = None
        for modname in ('EyeLinkCoreGraphicsPsychoPy', 'psychopy_eyelink_coregraphics'):
            try:
                mod = __import__(modname)
                CoreGraphics = getattr(mod, 'EyeLinkCoreGraphicsPsychoPy')
                break
            except (ImportError, AttributeError):
                continue
        if CoreGraphics is None:
            print('WARNING: EyeLinkCoreGraphicsPsychoPy not found (copy the .py next '
                  'to the script or `pip install psychopy-eyelink-coregraphics`) '
                  '-> calibration screen unavailable.')
        else:
            self.genv = CoreGraphics(self.tracker, win)
            self.genv.setCalibrationColors(params['cross_color'], params['bg_color'])
            pylink.openGraphicsEx(self.genv)

    # ------------------------------------------------------------------
    def calibrate(self):
        """Camera setup + calibration screen (press Enter/Esc to leave)."""
        if not HAVE_PYLINK or self.genv is None:
            return
        try:
            self.tracker.doTrackerSetup()
        except RuntimeError as err:
            print('Calibration error:', err)
            self.tracker.exitCalibration()

    def start_recording(self):
        if not HAVE_PYLINK:
            return
        self.tracker.setOfflineMode()
        # record samples + events to file and over the link
        err = self.tracker.startRecording(1, 1, 1, 1)
        if err:
            raise RuntimeError('startRecording failed with code %s' % err)
        pylink.pumpDelay(100)          # let the tracker settle
        self.recording = True

    def stop_recording(self):
        if not HAVE_PYLINK or not self.recording:
            return
        pylink.pumpDelay(100)
        self.tracker.stopRecording()
        self.recording = False

    def message(self, text):
        """Send a MSG to the EDF; also mirrors it to the CSV log."""
        if HAVE_PYLINK:
            self.tracker.sendMessage(text)
        self.log.edf_message(text)

    def close(self, receive_to):
        if not HAVE_PYLINK:
            return
        try:
            if self.recording:
                self.stop_recording()
            self.tracker.setOfflineMode()
            pylink.pumpDelay(500)
            self.tracker.closeDataFile()
            if not self.dummy:
                local = os.path.join(receive_to, self.edf_name)
                print('Receiving EDF -> %s' % local)
                self.tracker.receiveDataFile(self.edf_name, local)
        finally:
            self.tracker.close()
            if self.genv is not None:
                pylink.closeGraphics()


# --------------------------------------------------------------------------
# Cedrus response box (optional, via pyxid2 - bundled with PsychoPy Standalone)
# --------------------------------------------------------------------------
try:
    import pyxid2
    HAVE_PYXID = True
except ImportError:
    pyxid2 = None
    HAVE_PYXID = False


class ResponseBox(object):
    """Any-button detection on the first Cedrus XID device found.

    If pyxid2 is missing or no box is connected, `pressed()` always returns
    None and the keyboard remains the only way to confirm."""

    def __init__(self, enabled=True):
        self.dev = None
        if not enabled:
            return
        if not HAVE_PYXID:
            print('Cedrus: pyxid2 not available -> keyboard only.')
            return
        try:
            devices = pyxid2.get_xid_devices()
        except Exception as err:
            print('Cedrus: device scan failed (%s) -> keyboard only.' % err)
            return
        if not devices:
            print('Cedrus: no response box found -> keyboard only.')
            return
        self.dev = devices[0]
        try:                                   # pyxid2 >= 1.0.5
            self.dev.reset_timer()
        except AttributeError:                 # older pyxid2
            try:
                self.dev.reset_base_timer()
                self.dev.reset_rt_timer()
            except Exception:
                pass
        except Exception:
            pass
        print('Cedrus: using %s' % self.dev)

    def clear(self):
        """Drop any presses that happened before we started waiting."""
        if self.dev is None:
            return
        try:
            self.dev.poll_for_response()
            while self.dev.response_queue_size() > 0:
                self.dev.get_next_response()
                self.dev.poll_for_response()
            self.dev.clear_response_queue()
        except Exception:
            pass

    def pressed(self):
        """Return the key number of a fresh button press, or None."""
        if self.dev is None:
            return None
        try:
            self.dev.poll_for_response()
            while self.dev.response_queue_size() > 0:
                resp = self.dev.get_next_response()
                if resp.get('pressed'):
                    return resp.get('key')
        except Exception:
            return None
        return None


# --------------------------------------------------------------------------
# CSV event log
# --------------------------------------------------------------------------
class EventLog(object):
    FIELDS = ['participant', 'session', 'block', 'trial', 'event', 'stimulus',
              'size_frac', 't_session', 't_abs', 'planned_s', 'actual_s',
              'edf_message']

    def __init__(self, path, participant, session, clock):
        self.participant = participant
        self.session = session
        self.clock = clock
        self.fh = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.fh, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.pending_msg = ''

    def edf_message(self, text):
        self.pending_msg = text

    def write(self, event_name, block='', trial='', stimulus='', size_frac='',
              planned=None, actual=None, t=None):
        row = dict(
            participant=self.participant, session=self.session,
            block=block, trial=trial, event=event_name, stimulus=stimulus,
            size_frac='' if size_frac in ('', None) else '%.4f' % size_frac,
            t_session='%.4f' % (self.clock.getTime() if t is None else t),
            t_abs='%.4f' % core.getTime(),
            planned_s='' if planned is None else '%.3f' % planned,
            actual_s='' if actual is None else '%.4f' % actual,
            edf_message=self.pending_msg,
        )
        self.pending_msg = ''
        self.writer.writerow(row)
        self.fh.flush()

    def close(self):
        self.fh.close()


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------
class AbortExperiment(Exception):
    pass


class PLRTask(object):

    def __init__(self, params, info):
        self.p = params
        self.info = info
        self.rng = random.Random(info['seed'])
        self.session_clock = core.Clock()
        t_start = core.getTime()

        def step(label):
            """Print how long the previous startup step took."""
            now = core.getTime()
            print('  [%6.2f s] %s' % (now - step.last, label))
            step.last = now
        step.last = t_start

        # --- window ----------------------------------------------------
        self.win = visual.Window(
            fullscr=params['fullscreen'], screen=params['screen'],
            color=params['bg_color'], colorSpace='rgb', units='height',
            allowGUI=False, waitBlanking=True)
        self.win.mouseVisible = False
        step('window opened')

        # --- refresh rate & frame counts -------------------------------
        if params['measure_refresh']:
            hz = self.win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                             nWarmUpFrames=10, threshold=2)
            step('refresh rate measured')
        else:
            hz = None
        if hz is None or not (20 < hz < 500):
            if params['measure_refresh']:
                print('WARNING: could not measure refresh rate, assuming %.1f Hz'
                      % params['expected_hz'])
            hz = params['expected_hz']
        self.hz = hz
        self.frame_s = 1.0 / hz
        self.stim_frames = int(round(params['stim_s'] * hz))
        print('Refresh rate %.2f Hz -> stimulus = %d frames (%.1f ms)'
              % (hz, self.stim_frames, self.stim_frames * self.frame_s * 1000))

        # --- stimuli ---------------------------------------------------
        a = params['cross_size_frac']
        self.cross = visual.ShapeStim(
            self.win, vertices=[(-a, 0), (a, 0), (0, 0), (0, a), (0, -a)],
            closeShape=False, lineWidth=params['cross_line_px'],
            lineColor=params['cross_color'], colorSpace='rgb', units='height')
        self.discs = {}
        for name, frac in params['stimuli']:
            if frac is None:
                stim = visual.Rect(self.win, units='norm', size=(2, 2), pos=(0, 0),
                                   fillColor=params['stim_color'],
                                   lineColor=None, colorSpace='rgb')
            else:
                stim = visual.Circle(self.win, units='height', radius=frac / 2.0,
                                     pos=(0, 0), edges=256,
                                     fillColor=params['stim_color'],
                                     lineColor=None, colorSpace='rgb')
            self.discs[name] = stim
        self.text = visual.TextStim(self.win, text='', color=params['cross_color'],
                                    height=0.035, wrapWidth=1.4, units='height')
        step('stimuli built')

        # --- logging, response box, tracker ----------------------------
        self.log = EventLog(info['csv_path'], info['participant'], info['session'],
                            self.session_clock)
        self.box = ResponseBox(enabled=params['use_cedrus'])
        step('response box checked')
        self.el = EyeLinkSession(info['tracker_ip'], info['edf_name'], self.win,
                                 params, self.log)
        step('tracker ready')
        print('Startup total: %.2f s' % (core.getTime() - t_start))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def check_abort(self):
        if event.getKeys(keyList=[self.p['abort_key']]):
            raise AbortExperiment()

    def flip(self, *stims):
        """Draw the given stimuli and flip; return flip time on the session clock."""
        for s in stims:
            s.draw()
        self.win.flip()
        return self.session_clock.getTime()

    def hold(self, duration, *stims):
        """Keep `stims` on screen for `duration` seconds (frame loop, abortable).
        Returns the actual duration (time until the loop was left)."""
        t0 = self.session_clock.getTime()
        # keep flipping so that keys are polled and timing stays frame-locked;
        # stop when the *next* flip would fall after the deadline
        while self.session_clock.getTime() + self.frame_s / 2.0 < t0 + duration:
            self.check_abort()
            for s in stims:
                s.draw()
            self.win.flip()
        return self.session_clock.getTime() - t0

    def wait_confirm(self, *stims):
        """Show `stims` until a confirm key (space) or ANY Cedrus button is
        pressed.  Esc aborts.  Returns 'key:<name>' or 'cedrus:<n>'."""
        event.clearEvents()
        self.box.clear()
        while True:
            for s in stims:
                s.draw()
            self.win.flip()
            keys = event.getKeys(keyList=self.p['confirm_keys'] + [self.p['abort_key']])
            if self.p['abort_key'] in keys:
                raise AbortExperiment()
            if keys:
                return 'key:%s' % keys[0]
            btn = self.box.pressed()
            if btn is not None:
                return 'cedrus:%s' % btn

    def scaled(self, seconds):
        return seconds / float(self.p['debug_speed'])

    def uniform(self, rng):
        return self.rng.uniform(*rng)

    # ------------------------------------------------------------------
    # screens
    # ------------------------------------------------------------------
    def instructions(self):
        self.text.text = (
            'Během celého měření se prosím dívejte na šedý křížek\n'
            'uprostřed obrazovky. Občas se objeví krátký záblesk;\n'
            'snažte se v tu chvíli nemrkat a dál se dívejte do středu.\n\n'
            'Mezi bloky bude obrazovka na dvě minuty úplně černá -\n'
            'zůstaňte prosím v klidu a dívejte se před sebe.\n\n'
            'Pokračujte stisknutím tlačítka.')
        how = self.wait_confirm(self.text)
        self.log.write('instructions_confirmed', stimulus=how)
        self.flip()

    def end_screen(self):
        self.text.text = 'Konec měření. Děkujeme.'
        self.flip(self.text)
        core.wait(3.0)

    # ------------------------------------------------------------------
    # protocol
    # ------------------------------------------------------------------
    def run(self):
        p, el, log = self.p, self.el, self.log
        t0 = core.getTime()
        el.calibrate()
        print('  [%6.2f s] calibration finished' % (core.getTime() - t0))
        self.win.mouseVisible = False
        self.flip()
        self.instructions()

        el.start_recording()
        self.session_clock.reset()
        el.message('SESSION_START %s %s' % (self.info['participant'], self.info['session']))
        log.write('session_start')
        log.write('info_refresh_hz', actual=self.hz)
        log.write('info_seed', stimulus=str(self.info['seed']))

        try:
            for b in range(1, p['n_blocks'] + 1):
                self.run_block(b)
            el.message('SESSION_END')
            log.write('session_end')
            self.end_screen()
        except AbortExperiment:
            el.message('SESSION_ABORT')
            log.write('session_abort')
            print('Aborted by experimenter (%s).' % p['abort_key'])
        finally:
            self.shutdown()

    def run_block(self, b):
        p, el, log = self.p, self.el, self.log
        el.message('BLOCK_START %d' % b)
        log.write('block_start', block=b)

        # ---- pause: black screen -----------------------------------
        planned = self.scaled(p['pause_s'])
        self.flip()
        el.message('PAUSE_ON %d' % b)
        log.write('pause_on', block=b, planned=planned)
        actual = self.hold(planned)

        # ---- baseline: cross only ----------------------------------
        planned_bl = self.scaled(self.uniform(p['baseline_range_s']))
        self.flip(self.cross)
        el.message('PAUSE_OFF %d' % b)
        log.write('pause_off', block=b, actual=actual)
        el.message('CROSS_ON %d baseline' % b)
        log.write('cross_on', block=b, stimulus='baseline', planned=planned_bl)
        actual = self.hold(planned_bl, self.cross)

        # ---- four stimuli, each followed by an ISI with the cross ----
        for t, (name, frac) in enumerate(p['stimuli'], start=1):
            stim = self.discs[name]
            phase_before = 'baseline' if t == 1 else 'isi'
            size_frac = 1.0 if frac is None else frac

            el.message('TRIALID %d %d' % (b, t))
            log.write('trialid', block=b, trial=t, stimulus=name, size_frac=size_frac)
            # stimulus onset (first frame)
            t_on = self.flip(stim)
            el.message('CROSS_OFF %d %s' % (b, phase_before))
            log.write('cross_off', block=b, stimulus=phase_before, actual=actual, t=t_on)
            el.message('STIM_ON %d %d %s %.4f' % (b, t, name, size_frac))
            log.write('stim_on', block=b, trial=t, stimulus=name, size_frac=size_frac,
                      planned=p['stim_s'], t=t_on)
            # remaining frames of the stimulus
            for _ in range(self.stim_frames - 1):
                stim.draw()
                self.win.flip()
            # stimulus offset = cross back on
            planned_isi = self.scaled(self.uniform(p['isi_range_s']))
            t_off = self.flip(self.cross)
            el.message('STIM_OFF %d %d %s' % (b, t, name))
            log.write('stim_off', block=b, trial=t, stimulus=name, size_frac=size_frac,
                      actual=t_off - t_on, t=t_off)
            el.message('CROSS_ON %d isi' % b)
            log.write('cross_on', block=b, trial=t, stimulus='isi', planned=planned_isi,
                      t=t_off)
            actual = self.hold(planned_isi, self.cross)

        # ---- block end: cross off (next pause or end screen follows) ---
        self.flip()
        el.message('CROSS_OFF %d isi' % b)
        log.write('cross_off', block=b, trial=len(p['stimuli']), stimulus='isi',
                  actual=actual)
        el.message('BLOCK_END %d' % b)
        log.write('block_end', block=b)

    def shutdown(self):
        try:
            self.el.close(receive_to=DATA_DIR)
        except Exception as err:      # never lose the CSV because of the tracker
            print('Tracker shutdown error:', err)
        self.log.close()
        self.win.close()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def make_edf_name(participant, session):
    """EDF names on the Host PC: max 8 chars, letters/digits/underscore."""
    base = re.sub(r'[^A-Za-z0-9]', '', participant).upper()
    suffix = re.sub(r'[^A-Za-z0-9]', '', str(session)).upper()
    name = (base[:8 - len(suffix)] + suffix) if suffix else base
    if not name:
        name = 'PLRT'
    return name[:8] + '.EDF'


def main():
    info = {
        'participant': 'P00',
        'session': '1',
        'tracker_ip': PARAMS['tracker_ip'],
        'fullscreen': PARAMS['fullscreen'],
    }
    dlg = gui.DlgFromDict(
        info, title='PLRT',
        order=['participant', 'session', 'tracker_ip', 'fullscreen'],
        tip={'tracker_ip': 'Prázdné = dummy mode (bez trackeru)'})
    if not dlg.OK:
        core.quit()

    PARAMS['fullscreen'] = bool(info['fullscreen'])
    info['tracker_ip'] = info['tracker_ip'].strip()
    info['seed'] = PARAMS['seed'] if PARAMS['seed'] is not None else int(time.time())

    os.makedirs(DATA_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_pid = re.sub(r'[^A-Za-z0-9_-]', '', info['participant']) or 'P00'
    info['csv_path'] = os.path.join(
        DATA_DIR, '%s_%s_%s_plrt.csv' % (safe_pid, info['session'], stamp))
    info['edf_name'] = make_edf_name(info['participant'], info['session'])

    print('CSV : %s' % info['csv_path'])
    print('EDF : %s' % info['edf_name'])
    print('Seed: %s' % info['seed'])

    task = PLRTask(PARAMS, info)
    task.run()
    core.quit()


if __name__ == '__main__':
    main()
