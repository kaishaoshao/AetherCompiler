"""
Runtime: Command Buffer & Timeline Semaphore Architecture (借鉴 IREE HAL / Stream 方言).

Key Architectural Innovations (IREE 核心优势):
1. Asynchronous Timeline Semaphores (hal.semaphore):
   - Monotonically increasing 64-bit integer values tracking hardware execution progress.
   - Eliminates CPU-device sync stalls: CPU queues commands far ahead, device executes along the timeline.
2. Non-blocking Command Buffer Recording (hal.command_buffer):
   - Host records dispatch commands, memory bindings, and barriers into a binary command buffer.
   - Direct zero-overhead submission to asynchronous device queues.
"""

from typing import List, Dict, Any, Tuple, Optional
import time

class TimelineSemaphore:
    """
    Timeline Semaphore (借鉴 IREE hal.semaphore & Vulkan Timeline Semaphore).
    Tracks monotonic execution progress along a concurrent timeline.
    """
    def __init__(self, name: str, initial_value: int = 0):
        self.name = name
        self.current_value = initial_value

    def signal(self, value: int):
        assert value > self.current_value, f"Timeline semaphore must advance monotonically: {value} > {self.current_value}"
        self.current_value = value

    def is_signaled(self, target_value: int) -> bool:
        return self.current_value >= target_value

    def __repr__(self) -> str:
        return f"Semaphore('{self.name}', value={self.current_value})"


class CommandBuffer:
    """
    Command Buffer (借鉴 IREE hal.command_buffer).
    Stores a recorded linear command stream ready for asynchronous device submission.
    """
    def __init__(self, name: str):
        self.name = name
        self.commands: List[Dict[str, Any]] = []

    def bind_buffers(self, slot_to_dram_offsets: Dict[int, int]):
        """Binds DRAM physical offsets to local kernel argument slots."""
        self.commands.append({
            "type": "BIND_BUFFERS",
            "slots": slot_to_dram_offsets
        })

    def dispatch(self, executable_name: str, grid: Tuple[int, int, int]):
        """Dispatches an executable across a multi-dimensional workgroup grid."""
        self.commands.append({
            "type": "DISPATCH",
            "executable": executable_name,
            "grid": grid
        })

    def execution_barrier(
        self,
        wait_semaphore: Optional[TimelineSemaphore] = None,
        wait_value: int = 0,
        signal_semaphore: Optional[TimelineSemaphore] = None,
        signal_value: int = 0
    ):
        """Hardware execution barrier synchronizing asynchronous queues."""
        self.commands.append({
            "type": "BARRIER",
            "wait_sem": wait_semaphore,
            "wait_val": wait_value,
            "signal_sem": signal_semaphore,
            "signal_val": signal_value
        })

    def dma_copy_async(self, dst_offset: int, src_offset: int, num_bytes: int):
        """Asynchronous DMA transfer between memory domains."""
        self.commands.append({
            "type": "DMA_ASYNC",
            "dst": dst_offset,
            "src": src_offset,
            "bytes": num_bytes
        })

    def print_buffer(self):
        print(f"==================== IREE HAL Command Buffer [{self.name}] ====================")
        print(f"Total Recorded Commands: {len(self.commands)}")
        for idx, cmd in enumerate(self.commands[:15]):
            cmd_type = cmd["type"]
            if cmd_type == "BIND_BUFFERS":
                slots_str = ", ".join([f"slot[{k}]=0x{v:06X}" for k, v in cmd['slots'].items()])
                print(f"  [{idx:03d}] HAL.CMD.BIND_BUFFERS: {slots_str}")
            elif cmd_type == "DISPATCH":
                print(f"  [{idx:03d}] HAL.CMD.DISPATCH: @{cmd['executable']} Grid={cmd['grid']}")
            elif cmd_type == "BARRIER":
                w = f"Wait({cmd['wait_sem'].name}>={cmd['wait_val']})" if cmd['wait_sem'] else "None"
                s = f"Signal({cmd['signal_sem'].name}->{cmd['signal_val']})" if cmd['signal_sem'] else "None"
                print(f"  [{idx:03d}] HAL.CMD.BARRIER: {w} -> {s}")
            elif cmd_type == "DMA_ASYNC":
                print(f"  [{idx:03d}] HAL.CMD.DMA_ASYNC: dst=0x{cmd['dst']:06X}, src=0x{cmd['src']:06X}, {cmd['bytes']} B")
        if len(self.commands) > 15:
            print(f"  ... ({len(self.commands) - 15} more recorded hardware commands)")
        print("=================================================================================\n")


class AsyncDeviceQueue:
    """
    Asynchronous Device Execution Queue (借鉴 IREE HAL Device Queue).
    Executes command buffers asynchronously along the Timeline Semaphore.
    """
    def __init__(self, name: str = "npu_queue_0"):
        self.name = name
        self.timeline = TimelineSemaphore(f"{name}_timeline", initial_value=0)

    def submit(self, command_buffer: CommandBuffer, signal_on_complete: int):
        """
        Submits command buffer to the device queue.
        Advances timeline semaphore as work completes asynchronously.
        """
        for cmd in command_buffer.commands:
            if cmd["type"] == "BARRIER":
                wait_sem = cmd["wait_sem"]
                wait_val = cmd["wait_val"]
                if wait_sem and not wait_sem.is_signaled(wait_val):
                    raise RuntimeError(f"Queue stalled: waiting for {wait_sem.name} to reach {wait_val}")
                if cmd["signal_sem"]:
                    cmd["signal_sem"].signal(cmd["signal_val"])

        self.timeline.signal(signal_on_complete)
