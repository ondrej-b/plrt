#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plrt.py -- Pupillary Light Reflex Task (PLRT)
======================================================
taVNS project, September 2026

PsychoPy (coder-style) implementation of the PLR task.

How it fits the lab setup (PARAMS: use_eyelink=False, the default)
------------------------------------------------------------------
The EyeLink is not driven from here.  Its Host PC streams over SR Research's
own protocol to a separate Ubuntu machine, which republishes the eye data as
an LSL stream; LabRecorder stores that stream together with the marker stream
this script publishes, so everything ends up in one XDF with timestamps in a
common clock domain.  Calibration is likewise handled externally.  This script
therefore only presents the stimuli, publishes markers over LSL and writes its
own CSV event log.

Setting use_eyelink=True switches on the built-in pylink path instead: the
script then connects to a Host PC, opens an EDF, sends the same markers as MSG
records, starts/stops recording and pulls the EDF at the end.

Design (all durations configurable in PARAMS below)
---------------------------------------------------
Session:
    dialog -> open LSL outlet (-> optional tracker connection)
    -> wait for a recorder to subscribe -> instructions
    -> DARK ADAPTATION, black screen, 90 s
    -> 4 blocks, running straight on from one another
    -> close.

Trial (4 per block, stimulus order fixed 1 -> 4, 4 blocks):
    GET-READY   black screen + grey fixation cross     3 s
    STIMULUS    white disc or full white screen        200 ms   (cross off)
    ANALYSIS    black screen + grey fixation cross     15 s
    ISI         black screen, no cross                 U(5, 7) s

    stimulus 1  white disc, diameter 1/8 screen height
    stimulus 2  white disc, diameter 1/4 screen height
    stimulus 3  white disc, diameter 1/2 screen height
    stimulus 4  full white screen

Stimulus durations are frame-counted (12 frames at 60 Hz); the long
intervals are clock-based but keep flipping every frame so that the
keyboard (Esc = abort) stays responsive.

Markers (sent right after the flip that made the event visible; they go to the
LSL outlet always, and additionally into the EDF when use_eyelink=True):
    SESSION_START <participant> <session>
    ADAPT_ON                   ADAPT_OFF
    BLOCK_START <k>            BLOCK_END <k>
    TRIALID <k> <t>            (t = 1..4 within block)
    READY_ON <k> <t>           READY_OFF <k> <t>
    STIM_ON <k> <t> <name> <size_frac>
    STIM_OFF <k> <t> <name>
    ANALYSIS_ON <k> <t>        ANALYSIS_OFF <k> <t>
    ISI_ON <k> <t>             ISI_OFF <k> <t>
    SESSION_END | SESSION_ABORT
Every visible change of the display is thus bracketed by an ON/OFF pair; the
OFF of one screen and the ON of the next carry the same timestamp, because
they describe one and the same flip.

The same markers are published on a Lab Streaming Layer outlet (name 'PLRT',
type 'Markers', irregular rate, two string channels):
    channel 0 'text'  the exact string above
    channel 1 'json'  {"event":"STIM_ON","block":4,"trial":2,...}
LabRecorder records it alongside the eye / EEG / taVNS streams; the CSV column
`t_lsl` holds the LSL timestamp of each marker, which links the records.

Outputs (folder ``data/`` next to this script):
    <participant>_<session>_<date>_plrt.csv   event log from PsychoPy clocks
    <EDFNAME>.EDF                              only when use_eyelink=True

Run from a real terminal or the PsychoPy Runner, not from an inline IPython
console (see project notes on Spyder).
"""

import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime

from psychopy import core, event, gui, monitors, visual

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
PARAMS = dict(
    n_blocks=4,
    adapt_s=90.0,                  # dark adaptation, black screen, once per session
    ready_s=3.0,                   # get-ready cross before every stimulus
    stim_s=0.200,                  # each stimulus
    analysis_s=15.0,               # cross after the stimulus (pupil response)
    isi_range_s=(5.0, 7.0),        # black screen, no cross, end of every trial
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
    fullscreen=True,
    # Where to put the stimulus window, in order of preference:
    #   (monitor calibration name, screen index)
    # The first entry whose screen actually exists AND whose calibration is
    # defined on this machine wins.  Screen 0 is the primary display as Windows
    # reports it; the startup log lists the screens it found with their sizes.
    # Calibrations are registered once per machine by setup_monitor.py (see the
    # tavns-cpt repository; the SART task reads the same records).  None of this
    # affects stimulus size - those are fractions of the window height - it only
    # decides which display is used and what geometry is recorded with the data.
    monitor_priority=[
        ('taVNS_lab_ext', 1),      # external display in the testing booth
        ('taVNS_lab', 0),          # laptop panel / single-display machine
    ],
    expected_hz=60.0,              # used if measurement is off or unreliable
    measure_refresh=True,          # False -> trust expected_hz, skip the ~2 s test
    # --- EyeLink ------------------------------------------------------
    # In this lab the tracker is NOT driven from here: the Host PC streams to a
    # separate Ubuntu machine which republishes the eye data as its own LSL
    # stream, and LabRecorder stores that together with the markers below.  So
    # pylink stays switched off and the IP field is hidden from the dialog.
    # Set use_eyelink=True to connect to a Host PC directly (opens an EDF,
    # sends the same markers as MSG, starts/stops recording, pulls the EDF).
    use_eyelink=False,
    tracker_ip='100.1.1.1',        # only used when use_eyelink=True; '' -> dummy
    dummy_use_pylink=False,        # dummy mode: True exercises pylink's own dummy
                                   # connection (slow) incl. the calibration screen;
                                   # False bypasses pylink entirely (instant start)
    sample_rate=500,
    calibration_type='HV5',
    seed=None,                     # None -> random seed derived from time (logged)
    debug_speed=1.0,               # >1 shortens adaptation/ready/analysis/ISI for
                                   # testing; the 200 ms stimuli stay untouched
    abort_key='escape',
    confirm_keys=['space'],        # keyboard keys that confirm a screen ...
    use_cedrus=True,               # ... plus ANY button on a Cedrus response box (pyxid2)
    # --- Lab Streaming Layer (marker outlet) ---------------------------
    use_lsl=True,
    lsl_name='PLRT',               # stream name seen by LabRecorder
    lsl_type='Markers',
    lsl_wait_for_consumer=True,    # hold the start until a recorder subscribes
    lsl_wait_skip_key='s',         # ... or until this key is pressed
)

# --------------------------------------------------------------------------
# Instructions: one screen per item, each advanced by any confirm key or any
# Cedrus button.  '{minutes}' is filled in from the durations in PARAMS.
# --------------------------------------------------------------------------
INSTRUCTION_PAGES = [
    '=== PLRT ===\n\n'
    'Vítejte v experimentu měření pupilární reakce na světlo\n\n'
    'Během měření budeme snímat velikost vaší zornice.\n'
    'Vaším úkolem je pouze sedět v klidu a dívat se na obrazovku;\n'
    'nic nemusíte mačkat ani nijak odpovídat.\n\n'
    'Pokračujte stisknutím tlačítka.',

    'Když je na obrazovce křížek, snažte se hledět přímo na něj\n'
    'a nemrkat.\n\n'
    'Když je obrazovka prázdná, můžete dát očím odpočinout.\n\n'
    'Pokračujte stisknutím tlačítka.',

    'Občas se objeví krátký záblesk. Snažte se v tu chvíli nemrkat\n'
    'a dál se dívat do středu obrazovky.\n\n'
    'Na začátku bude obrazovka delší dobu úplně černá -\n'
    'zůstaňte prosím v klidu a dívejte se před sebe.\n\n'
    'Celé měření trvá přibližně {minutes} minut.\n\n'
    'Stisknutím tlačítka měření začne.',
]

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

    def __init__(self, ip, edf_name, win, params, log, lsl=None):
        self.win = win
        self.params = params
        self.log = log
        self.lsl = lsl
        self.edf_name = edf_name
        self.recording = False
        self.dummy = (ip == '' or ip is None)
        self.genv = None
        self.uses_pylink = False       # True only once a real pylink object exists

        if not params['use_eyelink']:
            print('EyeLink: not driven from this script (eye data reach the '
                  'recording via the lab LSL stream).')
            self.tracker = DummyTracker()
            self.dummy = True
            return

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
        self.uses_pylink = True

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
        if not self.uses_pylink or self.genv is None:
            return
        try:
            self.tracker.doTrackerSetup()
        except RuntimeError as err:
            print('Calibration error:', err)
            self.tracker.exitCalibration()

    def start_recording(self):
        if not self.uses_pylink:
            return
        self.tracker.setOfflineMode()
        # record samples + events to file and over the link
        err = self.tracker.startRecording(1, 1, 1, 1)
        if err:
            raise RuntimeError('startRecording failed with code %s' % err)
        pylink.pumpDelay(100)          # let the tracker settle
        self.recording = True

    def stop_recording(self):
        if not self.uses_pylink or not self.recording:
            return
        pylink.pumpDelay(100)
        self.tracker.stopRecording()
        self.recording = False

    def send(self, text):
        """Write one MSG into the EDF. No-op unless a real tracker is in use.

        Deliberately does NOT touch LSL or the log: the marker has already gone
        out by the time this is called (see PLRTask.mark and run_block)."""
        if self.uses_pylink:
            self.tracker.sendMessage(text)

    def close(self, receive_to):
        if not self.uses_pylink:
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
# Lab Streaming Layer marker outlet (pylsl - bundled with PsychoPy Standalone)
# --------------------------------------------------------------------------
try:
    import pylsl
    HAVE_PYLSL = True
except ImportError:
    pylsl = None
    HAVE_PYLSL = False


# Argument names for each marker, used to build the JSON channel.  Extra
# arguments beyond this list are collected into "args".
MARKER_FIELDS = {
    'SESSION_START': ['participant', 'session'],
    'SESSION_END': [],
    'SESSION_ABORT': [],
    'ADAPT_ON': [],
    'ADAPT_OFF': [],
    'BLOCK_START': ['block'],
    'BLOCK_END': ['block'],
    'TRIALID': ['block', 'trial'],
    'READY_ON': ['block', 'trial'],
    'READY_OFF': ['block', 'trial'],
    'STIM_ON': ['block', 'trial', 'stim', 'size'],
    'STIM_OFF': ['block', 'trial', 'stim'],
    'ANALYSIS_ON': ['block', 'trial'],
    'ANALYSIS_OFF': ['block', 'trial'],
    'ISI_ON': ['block', 'trial'],
    'ISI_OFF': ['block', 'trial'],
}


def marker_to_dict(text):
    """'STIM_ON 4 2 circle_1_4 0.2500' -> dict, for the JSON channel."""
    parts = text.split()
    out = {'event': parts[0] if parts else ''}
    names = MARKER_FIELDS.get(out['event'], [])
    values = parts[1:]
    for i, value in enumerate(values):
        key = names[i] if i < len(names) else 'arg%d' % (i + 1)
        try:                       # keep numbers numeric in the JSON
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


class MarkerStream(object):
    """Two-channel string marker outlet: ['<plain text>', '<json>'].

    Channel 0 carries exactly the same string that goes into the EDF, so the
    two records can be matched line by line; channel 1 carries the parsed
    version for automated processing."""

    def __init__(self, params, info):
        self.outlet = None
        if not params['use_lsl']:
            return
        if not HAVE_PYLSL:
            print('WARNING: pylsl not available -> no LSL markers.')
            return
        source_id = 'PLRT_%s_%s' % (info['participant'], info['session'])
        try:
            sinfo = pylsl.StreamInfo(
                name=params['lsl_name'], type=params['lsl_type'],
                channel_count=2, nominal_srate=pylsl.IRREGULAR_RATE,
                channel_format='string', source_id=source_id)
            desc = sinfo.desc()
            desc.append_child_value('task', 'PLRT')
            desc.append_child_value('participant', str(info['participant']))
            desc.append_child_value('session', str(info['session']))
            desc.append_child_value('edf_file', info['edf_name'])
            desc.append_child_value('monitor', info.get('monitor_desc', 'undefined'))
            channels = desc.append_child('channels')
            for label in ('text', 'json'):
                ch = channels.append_child('channel')
                ch.append_child_value('label', label)
                ch.append_child_value('type', 'Marker')
                ch.append_child_value('format', 'string')
            self.outlet = pylsl.StreamOutlet(sinfo)
            print("LSL: outlet '%s' (type '%s', source_id '%s') open."
                  % (params['lsl_name'], params['lsl_type'], source_id))
        except Exception as err:
            print('WARNING: could not open the LSL outlet (%s) -> no LSL markers.'
                  % err)
            self.outlet = None

    @property
    def active(self):
        return self.outlet is not None

    # --- split push, for use around a flip ----------------------------
    # prepare() does the expensive part (string formatting, JSON encoding)
    # BEFORE the flip; push_at() is then a single library call, so the gap
    # between the physical frame onset and the marker is a few microseconds.
    def prepare(self, text):
        """Build the 2-channel sample ahead of time; None if the outlet is off."""
        if self.outlet is None:
            return None
        return [text, json.dumps(marker_to_dict(text), separators=(',', ':'))]

    def now(self):
        """Current LSL time, or '' when the outlet is off."""
        return pylsl.local_clock() if self.outlet is not None else ''

    def push_at(self, sample, timestamp):
        """Send a sample built by prepare(). Keep this call as short as possible."""
        if self.outlet is None or sample is None:
            return ''
        try:
            self.outlet.push_sample(sample, timestamp)
        except Exception as err:
            print('LSL push failed (%s): %s' % (err, sample[0]))
            return ''
        return timestamp

    def has_consumer(self):
        if self.outlet is None:
            return False
        try:
            return bool(self.outlet.have_consumers())
        except Exception:
            return False

    def close(self):
        self.outlet = None          # StreamOutlet is freed with the object


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
              'size_frac', 't_session', 't_abs', 't_lsl', 'planned_s',
              'actual_s', 'edf_message']

    def __init__(self, path, participant, session, clock):
        self.participant = participant
        self.session = session
        self.clock = clock
        self.fh = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.fh, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.fh.flush()
        self.pending_msg = ''
        self.pending_lsl = ''
        self.buffer = []           # rows wait here; disk I/O never near a flip

    def edf_message(self, text, lsl_timestamp=''):
        """Remember the marker so the next CSV row can carry it."""
        self.pending_msg = text
        self.pending_lsl = lsl_timestamp

    def write(self, event_name, block='', trial='', stimulus='', size_frac='',
              planned=None, actual=None, t=None):
        row = dict(
            participant=self.participant, session=self.session,
            block=block, trial=trial, event=event_name, stimulus=stimulus,
            size_frac='' if size_frac in ('', None) else '%.4f' % size_frac,
            t_session='%.4f' % (self.clock.getTime() if t is None else t),
            t_abs='%.4f' % core.getTime(),
            t_lsl='' if self.pending_lsl == '' else '%.6f' % self.pending_lsl,
            planned_s='' if planned is None else '%.3f' % planned,
            actual_s='' if actual is None else '%.4f' % actual,
            edf_message=self.pending_msg,
        )
        self.pending_msg = ''
        self.pending_lsl = ''
        self.buffer.append(row)

    def flush(self):
        """Write buffered rows out. Call only during quiet periods."""
        if not self.buffer:
            return
        for row in self.buffer:
            self.writer.writerow(row)
        self.buffer = []
        self.fh.flush()

    def close(self):
        self.flush()
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
        self._pending_off = None       # OFF marker owed by the screen on display
        self.last_actual = None        # measured duration of the phase just ended
        t_start = core.getTime()

        def step(label):
            """Print how long the previous startup step took."""
            now = core.getTime()
            print('  [%6.2f s] %s' % (now - step.last, label))
            step.last = now
        step.last = t_start

        # --- monitor and display ----------------------------------------
        self.mon, self.monitor_desc, self.screen_index = self.choose_monitor()
        if self.mon is not None:
            info['monitor_name'] = self.mon.name
            info['monitor_desc'] = self.monitor_desc
        step('monitor resolved')

        # --- window ----------------------------------------------------
        self.win = visual.Window(
            monitor=self.mon,
            fullscr=params['fullscreen'], screen=self.screen_index,
            color=params['bg_color'], colorSpace='rgb', units='height',
            allowGUI=False, waitBlanking=True)
        self.win.mouseVisible = False
        # What the window ENDED UP with.  PsychoPy logs "Monitor specification
        # not found. Creating a temporary one..." whenever anything builds a
        # nameless monitor, which can come from elsewhere in the same run, so
        # this line (and the CSV row below) is the authoritative record.
        self.window_monitor_desc = self.describe_window_monitor()
        print('Window monitor: %s on screen %d, window %sx%s px'
              % (self.window_monitor_desc, self.screen_index,
                 self.win.size[0], self.win.size[1]))
        self.check_resolution()
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
        self.lsl = MarkerStream(params, info)
        step('LSL outlet ready')
        self.el = EyeLinkSession(info['tracker_ip'], info['edf_name'], self.win,
                                 params, self.log, lsl=self.lsl)
        step('tracker ready')
        print('Startup total: %.2f s' % (core.getTime() - t_start))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def describe_window_monitor(self):
        """Name and geometry of the monitor the open window is really using."""
        mon = getattr(self.win, 'monitor', None)
        if mon is None:
            return 'none'
        try:
            name = mon.name
            width, dist, size = mon.getWidth(), mon.getDistance(), mon.getSizePix()
        except Exception as err:
            return 'unreadable (%s)' % err
        if width is None or dist is None:
            return '%s (bez geometrie)' % name
        return '%s %.1fcm %.1fcm %sx%s' % (name, width, dist,
                                           size[0] if size else '?',
                                           size[1] if size else '?')

    def check_resolution(self):
        """Warn if the calibration was measured at a different resolution.

        This is the check that catches a calibration belonging to the other
        display of a laptop-plus-external setup: the geometry would then be
        wrong even though a monitor was found."""
        if self.mon is None:
            return
        try:
            calib = self.mon.getSizePix()
            actual = [int(v) for v in self.win.size]
        except Exception:
            return
        if not calib or list(calib) == actual:
            return
        print("WARNING: monitor '%s' byl kalibrován na %sx%s px, ale okno je "
              "%sx%s px. Zkontrolujte, že kalibrace patří tomuto displeji "
              "(šířka v cm a vzdálenost by pak byly špatné)."
              % (self.mon.name, calib[0], calib[1], actual[0], actual[1]))

    @staticmethod
    def list_screens():
        """[(index, (w, h) or None), ...] for the displays a window can open on."""
        try:
            import pyglet
            screens = pyglet.canvas.get_display().get_screens()
            return [(i, (sc.width, sc.height)) for i, sc in enumerate(screens)]
        except Exception as err:
            print('WARNING: nelze vyjmenovat displeje (%s); predpokladam jeden.'
                  % err)
            return [(0, None)]

    def choose_monitor(self):
        """Pick the first usable (calibration, screen) pair from monitor_priority.

        Returns (Monitor or None, description, screen index).  Note that a
        PsychoPy 'Monitor' is only a named record of screen geometry - it does
        not select a display.  The display is the `screen` index below."""
        screens = self.list_screens()
        print('Displeje: %s' % ', '.join(
            'screen %d = %s' % (i, ('%dx%d' % wh) if wh else '?')
            for i, wh in screens))
        n_screens = len(screens)
        for name, screen_idx in self.p['monitor_priority']:
            if screen_idx >= n_screens:
                print("Monitor '%s': screen %d není k dispozici -> zkouším dál."
                      % (name, screen_idx))
                continue
            mon, desc = self.load_monitor(name, quiet=True)
            if mon is None:
                print("Monitor '%s': kalibrace není na tomto počítači -> "
                      "zkouším dál." % name)
                continue
            print('Použiji monitor %s na screen %d.' % (desc, screen_idx))
            return mon, desc, screen_idx
        # Nothing matched: say where we looked, then fall back to the primary
        # display with PsychoPy's defaults.
        print('WARNING: žádný z nakonfigurovaných monitorů není použitelný -> '
              'PsychoPy použije výchozí hodnoty a screen 0; do dat se nezapíše '
              'geometrie obrazovky.')
        self.load_monitor(self.p['monitor_priority'][0][0])   # prints details
        return None, 'undefined', 0

    @staticmethod
    def load_monitor(name, quiet=False):
        """Fetch the shared lab monitor definition.

        Returns (Monitor or None, description string).  A missing or incomplete
        definition only warns: sizes here are fractions of the window height, so
        the task itself does not need the geometry - but then nothing about the
        screen is recorded with the data."""
        advice = ("Spusťte setup_monitor.py (repozitář tavns-cpt) nebo monitor "
                  "nadefinujte v PsychoPy Monitor Center.")
        try:
            known = monitors.getAllMonitors()
        except Exception as err:
            print('WARNING: monitor list unavailable (%s).' % err)
            return None, 'unavailable'
        if name not in known:
            # Say WHERE we looked and WHAT is there: a monitor defined in the
            # Monitor Center of a different Windows account, an elevated shell
            # or another PsychoPy install lands in a different prefs folder and
            # is simply invisible here.
            folder = getattr(monitors, 'monitorFolder', None)
            if folder is None:
                try:
                    from psychopy import prefs
                    folder = os.path.join(prefs.paths['userPrefsDir'], 'monitors')
                except Exception:
                    folder = '<neznámá složka>'
            if not quiet:
                print("WARNING: monitor '%s' není na tomto počítači "
                      "nadefinovaný. %s" % (name, advice))
                print("         hledáno v: %s" % folder)
                print("         nalezené monitory: %s"
                      % (', '.join(known) if known else '(žádné)'))
            return None, 'undefined'
        mon = monitors.Monitor(name)
        width, dist, size = mon.getWidth(), mon.getDistance(), mon.getSizePix()
        if width is None or dist is None:
            print("WARNING: monitor '%s' nemá vyplněnou šířku nebo vzdálenost -> "
                  "geometrie se nezapíše. %s" % (name, advice))
            return None, 'incomplete'
        desc = '%s %.1fcm %.1fcm %sx%s' % (name, width, dist,
                                           size[0] if size else '?',
                                           size[1] if size else '?')
        print("Monitor '%s': šířka %.1f cm, vzdálenost %.1f cm, %s px."
              % (name, width, dist, size))
        return mon, desc

    def check_abort(self):
        if event.getKeys(keyList=[self.p['abort_key']]):
            raise AbortExperiment()

    def flip(self, *stims):
        """Draw the given stimuli and flip; return flip time on the session clock."""
        for s in stims:
            s.draw()
        self.win.flip()
        return self.session_clock.getTime()

    def mark(self, text, event_name, ts=None, **logkw):
        """Emit a marker that is NOT tied to a frame onset.

        Order is still LSL first, bookkeeping after, but the timing of these
        markers (block/pause boundaries) is not critical - the display change
        they describe is a slow one."""
        sample = self.lsl.prepare(text)
        if ts is None:
            ts = self.lsl.now()
        self.lsl.push_at(sample, ts)
        self.el.send(text)
        self.log.edf_message(text, ts)
        self.log.write(event_name, **logkw)
        return ts

    def hold(self, duration, *stims):
        """Keep `stims` on screen for `duration` seconds (frame loop, abortable).
        Returns the actual duration (time until the loop was left)."""
        t0 = self.session_clock.getTime()
        self.log.flush()           # disk I/O here, never around a stimulus flip
        # keep flipping so that keys are polled and timing stays frame-locked;
        # stop when the *next* flip would fall after the deadline
        while self.session_clock.getTime() + self.frame_s / 2.0 < t0 + duration:
            self.check_abort()
            for s in stims:
                s.draw()
            self.win.flip()
        self.last_actual = self.session_clock.getTime() - t0
        return self.last_actual

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

    # --- screen changes -----------------------------------------------
    # Each screen "arms" the OFF marker that will be sent when the next screen
    # replaces it, so every phase is bracketed by an ON/OFF pair and the two
    # markers around one flip share a timestamp.
    def arm_off(self, text, event_name, **logkw):
        self._pending_off = (text, event_name, logkw)

    def fire_off(self, ts, actual=None):
        if self._pending_off is None:
            return
        text, event_name, logkw = self._pending_off
        self._pending_off = None
        if actual is not None:
            logkw = dict(logkw, actual=actual)
        self.mark(text, event_name, ts=ts, **logkw)

    def show(self, stims, on_text, on_event, actual=None, **logkw):
        """Non-critical screen change: flip, close the previous phase, open
        the new one.  Both markers get the timestamp of this flip."""
        for s in stims:
            s.draw()
        self.win.flip()
        ts = self.lsl.now()
        self.fire_off(ts, actual=actual)
        self.mark(on_text, on_event, ts=ts, **logkw)
        return ts

    # ------------------------------------------------------------------
    # screens
    # ------------------------------------------------------------------
    def estimated_minutes(self):
        """Rough session length from the current parameters, for the text."""
        p = self.p
        per_trial = (p['ready_s'] + p['stim_s'] + p['analysis_s']
                     + sum(p['isi_range_s']) / 2.0)
        total = self.scaled(p['adapt_s'] + p['n_blocks'] * len(p['stimuli'])
                            * per_trial)
        return max(1, int(round(total / 60.0)))

    def instructions(self):
        """Three text screens, each advanced by a key or a Cedrus button."""
        pages = [page.format(minutes=self.estimated_minutes())
                 for page in INSTRUCTION_PAGES]
        for i, page in enumerate(pages, start=1):
            self.text.text = page
            how = self.wait_confirm(self.text)
            self.log.write('instructions_page', trial=i, stimulus=how)
        self.flip()

    def wait_for_recorder(self):
        """Hold the session until something subscribes to the LSL outlet.

        Esc aborts, the skip key continues without a recorder."""
        p = self.p
        if not (p['use_lsl'] and p['lsl_wait_for_consumer'] and self.lsl.active):
            return
        if self.lsl.has_consumer():
            print('LSL: recorder already connected.')
            return
        print('LSL: waiting for a recorder to subscribe '
              '(press %s to continue without one).' % p['lsl_wait_skip_key'])
        self.text.text = ('Čekám na připojení LSL recorderu...\n\n'
                          'Spusťte LabRecorder a zaškrtněte stream "%s".\n\n'
                          '(%s = pokračovat bez záznamu, Esc = konec)'
                          % (p['lsl_name'], p['lsl_wait_skip_key']))
        event.clearEvents()
        while not self.lsl.has_consumer():
            self.text.draw()
            self.win.flip()
            keys = event.getKeys(keyList=[p['lsl_wait_skip_key'],
                                          p['abort_key']])
            if p['abort_key'] in keys:
                raise AbortExperiment()
            if keys:
                print('LSL: continuing without a recorder.')
                self.log.write('lsl_wait_skipped')
                self.flip()
                return
        print('LSL: recorder connected.')
        self.log.write('lsl_consumer_connected')
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
        # Calibration is done externally (Host PC / separate procedure) and is
        # deliberately not driven from this script.  To bring it back, uncomment
        # the three lines below.
        # t0 = core.getTime()
        # el.calibrate()
        # print('  [%6.2f s] calibration finished' % (core.getTime() - t0))
        self.win.mouseVisible = False
        self.flip()
        self.wait_for_recorder()
        self.instructions()

        el.start_recording()
        self.session_clock.reset()
        self.mark('SESSION_START %s %s'
                  % (self.info['participant'], self.info['session']),
                  'session_start')
        log.write('info_refresh_hz', actual=self.hz)
        log.write('info_monitor', stimulus=self.monitor_desc)
        log.write('info_window_monitor', stimulus=self.window_monitor_desc)
        log.write('info_seed', stimulus=str(self.info['seed']))

        try:
            self.dark_adaptation()
            for b in range(1, p['n_blocks'] + 1):
                self.run_block(b)
            # close whatever screen is still up (the last ISI)
            ts = self.lsl.now()
            self.fire_off(ts, actual=self.last_actual)
            self.mark('SESSION_END', 'session_end', ts=ts)
            self.end_screen()
        except AbortExperiment:
            self.mark('SESSION_ABORT', 'session_abort')
            print('Aborted by experimenter (%s).' % p['abort_key'])
        finally:
            self.shutdown()

    def dark_adaptation(self):
        """Black screen at the start of the session, once."""
        planned = self.scaled(self.p['adapt_s'])
        self.show((), 'ADAPT_ON', 'adapt_on', planned=planned)
        self.arm_off('ADAPT_OFF', 'adapt_off')
        self.hold(planned)                        # black: nothing to draw

    def run_block(self, b):
        p, el, log, lsl = self.p, self.el, self.log, self.lsl
        self.mark('BLOCK_START %d' % b, 'block_start', block=b)

        for t, (name, frac) in enumerate(p['stimuli'], start=1):
            stim = self.discs[name]
            size_frac = 1.0 if frac is None else frac

            self.mark('TRIALID %d %d' % (b, t), 'trialid', block=b, trial=t,
                      stimulus=name, size_frac=size_frac)

            # ---------- GET-READY: cross ----------
            planned_ready = self.scaled(p['ready_s'])
            self.show((self.cross,), 'READY_ON %d %d' % (b, t), 'ready_on',
                      actual=self.last_actual, block=b, trial=t,
                      planned=planned_ready)
            self.arm_off('READY_OFF %d %d' % (b, t), 'ready_off',
                         block=b, trial=t)
            self.hold(planned_ready, self.cross)   # cross stays up for the whole phase

            # ---------- STIMULUS ONSET ----------
            # Everything that can be computed in advance is computed here, so
            # that the flip -> timestamp -> push sequence below contains no
            # string formatting, no JSON, no disk and no tracker call.
            msg_on = 'STIM_ON %d %d %s %.4f' % (b, t, name, size_frac)
            msg_ready_off = 'READY_OFF %d %d' % (b, t)
            smp_on = lsl.prepare(msg_on)
            smp_ready_off = lsl.prepare(msg_ready_off)
            planned_analysis = self.scaled(p['analysis_s'])
            msg_off = 'STIM_OFF %d %d %s' % (b, t, name)
            msg_analysis_on = 'ANALYSIS_ON %d %d' % (b, t)
            smp_off = lsl.prepare(msg_off)
            smp_analysis_on = lsl.prepare(msg_analysis_on)
            stim.draw()

            self.win.flip()                       # <- physical onset
            ts_on = lsl.now()                     # <- ~1 us later
            lsl.push_at(smp_on, ts_on)            # <- the critical marker

            # --- non-critical bookkeeping, all stamped with ts_on ---
            self._pending_off = None              # READY_OFF is sent right here
            t_on = self.session_clock.getTime()
            lsl.push_at(smp_ready_off, ts_on)
            el.send(msg_ready_off)
            el.send(msg_on)
            log.edf_message(msg_ready_off, ts_on)
            log.write('ready_off', block=b, trial=t, actual=self.last_actual,
                      t=t_on)
            log.edf_message(msg_on, ts_on)
            log.write('stim_on', block=b, trial=t, stimulus=name,
                      size_frac=size_frac, planned=p['stim_s'], t=t_on)

            # remaining frames of the stimulus
            for _ in range(self.stim_frames - 1):
                stim.draw()
                self.win.flip()

            # ---------- STIMULUS OFFSET -> ANALYSIS: cross ----------
            self.cross.draw()
            self.win.flip()                       # <- physical offset
            ts_off = lsl.now()
            lsl.push_at(smp_off, ts_off)          # <- the critical marker

            t_off = self.session_clock.getTime()
            lsl.push_at(smp_analysis_on, ts_off)
            el.send(msg_off)
            el.send(msg_analysis_on)
            log.edf_message(msg_off, ts_off)
            log.write('stim_off', block=b, trial=t, stimulus=name,
                      size_frac=size_frac, actual=t_off - t_on, t=t_off)
            log.edf_message(msg_analysis_on, ts_off)
            log.write('analysis_on', block=b, trial=t, planned=planned_analysis,
                      t=t_off)
            self.arm_off('ANALYSIS_OFF %d %d' % (b, t), 'analysis_off',
                         block=b, trial=t)
            self.hold(planned_analysis, self.cross)  # cross stays up for the whole phase

            # ---------- ISI: black screen, no cross ----------
            planned_isi = self.scaled(self.uniform(p['isi_range_s']))
            self.show((), 'ISI_ON %d %d' % (b, t), 'isi_on',
                      actual=self.last_actual, block=b, trial=t,
                      planned=planned_isi)
            self.arm_off('ISI_OFF %d %d' % (b, t), 'isi_off', block=b, trial=t)
            self.hold(planned_isi)                # black: nothing to draw

        self.mark('BLOCK_END %d' % b, 'block_end', block=b)

    def shutdown(self):
        try:
            self.el.close(receive_to=DATA_DIR)
        except Exception as err:      # never lose the CSV because of the tracker
            print('Tracker shutdown error:', err)
        self.lsl.close()
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
    # The dict keys are what the dialog shows as field labels, so they are
    # spelled the way the experimenter should read them; they are copied onto
    # the internal names right after the dialog closes.
    fields = {
        'participant_id': 'P00',
        'session_id': 'S00',
        'fullscreen': PARAMS['fullscreen'],
    }
    order = ['participant_id', 'session_id', 'fullscreen']
    tips = {}
    if PARAMS['use_eyelink']:          # only ask for the IP if we talk to a tracker
        fields['tracker_ip'] = PARAMS['tracker_ip']
        order.insert(2, 'tracker_ip')
        tips['tracker_ip'] = 'IP adresa EyeLink Host PC; prázdné = bez trackeru'
    dlg = gui.DlgFromDict(fields, title='PLRT', order=order, tip=tips)
    if not dlg.OK:
        core.quit()

    info = {
        'participant': str(fields['participant_id']).strip() or 'P00',
        'session': str(fields['session_id']).strip() or 'S00',
    }
    PARAMS['fullscreen'] = bool(fields['fullscreen'])
    info['tracker_ip'] = str(fields.get('tracker_ip', '')).strip()
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
