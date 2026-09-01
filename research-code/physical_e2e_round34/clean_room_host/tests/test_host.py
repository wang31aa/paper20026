import json
import tempfile
import unittest
from pathlib import Path

from clean_room_host import (
    Arm, EvidenceClass, Frame, HostOrchestrator, HostState, LoopbackTransport,
    MessageType, SafetyLatch,
)
from clean_room_host.hashlog import verify


def frame(**overrides):
    values = dict(
        protocol_version=1,
        message_type=MessageType.TELEMETRY_MIRROR,
        experiment_prefix="TEST_ONLY_NON_EVIDENTIARY",
        run_id=7,
        sender_node=1,
        receiver_nodes=(255,),
        arm=Arm.OBSERVER_LOOP,
        sequence=0,
        sender_tick=3,
        sender_time_us=0,
        payload={"test_marker": "NO_MEASUREMENT"},
    )
    values.update(overrides)
    return Frame(**values)


class HostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "TEST_ONLY.audit.jsonl"
        self.transport = LoopbackTransport()
        self.host = HostOrchestrator(
            self.transport, self.path, protocol_version=1,
            experiment_prefix="TEST_ONLY_NON_EVIDENTIARY",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def start(self):
        self.host.self_test_passed()
        self.host.verify_configuration(run_id=7, arm=Arm.OBSERVER_LOOP,
                                       full_hashes_match=True, clocks_valid=True)
        self.host.arm_zero(operator_confirmed=True, all_heartbeats_valid=True,
                           guards_closed=True)
        self.host.start(start_tick=3)

    def test_default_safe_zero_and_non_evidentiary_label(self):
        self.assertEqual(self.transport.safe_zero_calls, 1)
        first = json.loads(self.path.read_text().splitlines()[0])
        self.assertEqual(first["event"]["evidence_class"],
                         EvidenceClass.NON_EVIDENTIARY.value)

    def test_valid_mirror_frame_and_hash_chain(self):
        self.start()
        self.host.admit(frame())
        self.assertEqual(self.host.state, HostState.RUNNING)
        self.assertEqual(verify(self.path), self.host.log.head)

    def test_acl_violation_latches_zero(self):
        self.start()
        forbidden = frame(message_type=MessageType.LEADER_SAMPLE,
                          sender_node=0, receiver_nodes=(2,))
        with self.assertRaises(SafetyLatch):
            self.host.admit(forbidden)
        self.assertEqual(self.host.state, HostState.SAFE_ZERO_LATCHED)
        self.assertEqual(self.transport.safe_zero_calls, 2)

    def test_future_tick_latches_zero(self):
        self.start()
        with self.assertRaises(SafetyLatch):
            self.host.admit(frame(sender_tick=4))

    def test_repeated_sequence_latches_zero(self):
        self.start()
        self.host.admit(frame())
        with self.assertRaises(SafetyLatch):
            self.host.admit(frame())

    def test_invalid_guard_latches_zero(self):
        self.host.self_test_passed()
        self.host.verify_configuration(run_id=7, arm=Arm.OBSERVER_LOOP,
                                       full_hashes_match=True, clocks_valid=True)
        with self.assertRaises(SafetyLatch):
            self.host.arm_zero(operator_confirmed=True, all_heartbeats_valid=True,
                               guards_closed=False)

    def test_log_is_new_file_only_and_tamper_detected(self):
        self.host.self_test_passed()
        with self.assertRaises(FileExistsError):
            HostOrchestrator(self.transport, self.path, protocol_version=1,
                             experiment_prefix="TEST_ONLY_NON_EVIDENTIARY")
        lines = self.path.read_text().splitlines()
        record = json.loads(lines[0])
        record["event"]["event_type"] = "TAMPERED_TEST_MARKER"
        self.path.write_text(json.dumps(record) + "\n" + "\n".join(lines[1:]) + "\n")
        with self.assertRaises(ValueError):
            verify(self.path)


if __name__ == "__main__":
    unittest.main()
