from __future__ import annotations

from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_uint,
    c_ushort,
    create_string_buffer,
)
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence


MAX_AXIS = 3
DEFAULT_LIB_PATH = (
    Path(__file__).resolve().parent
    / "vendor"
    / "ubuntu"
    / "lib"
    / "64"
    / "libFMC4030-Lib.so"
)


class FMC4030Error(RuntimeError):
    """Raised when a controller API call fails."""


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2


class MachineStatus(Structure):
    _fields_ = [
        ("realPos", c_float * MAX_AXIS),
        ("realSpeed", c_float * MAX_AXIS),
        ("inputStatus", c_uint),
        ("outputStatus", c_uint),
        ("limitNStatus", c_uint),
        ("limitPStatus", c_uint),
        ("machineRunStatus", c_uint),
        ("axisStatus", c_uint * MAX_AXIS),
        ("homeStatus", c_uint),
        ("file", c_char * 600),
    ]


class MachineDeviceParams(Structure):
    _fields_ = [
        ("id", c_uint),
        ("bound232", c_uint),
        ("bound485", c_uint),
        ("ip", c_char * 15),
        ("port", c_int),
        ("div", c_int * MAX_AXIS),
        ("lead", c_int * MAX_AXIS),
        ("softLimitMax", c_int * MAX_AXIS),
        ("softLimitMin", c_int * MAX_AXIS),
        ("homeTime", c_int * MAX_AXIS),
    ]


class MachineVersion(Structure):
    _fields_ = [
        ("firmware", c_uint),
        ("lib", c_uint),
        ("serialnumber", c_uint),
    ]


@dataclass
class ControllerConfig:
    controller_id: int = 1
    ip: str = "192.168.0.30"
    port: int = 8088


@dataclass
class DeviceParameters:
    id: int
    bound232: int
    bound485: int
    ip: str
    port: int
    div: Sequence[int]
    lead: Sequence[int]
    soft_limit_max: Sequence[int]
    soft_limit_min: Sequence[int]
    home_time: Sequence[int]


@dataclass
class VersionInfo:
    firmware: int
    library: int
    serial: int


def _encode_str(text: str) -> bytes:
    return text.encode("utf-8")


def axis_mask(axes: Iterable[Axis]) -> int:
    mask = 0
    for axis in axes:
        mask |= 1 << int(axis)
    return mask


class FMC4030Controller:
    """ctypes wrapper around the Ubuntu FMC4030 shared library."""

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        path = Path(lib_path or DEFAULT_LIB_PATH)
        if not path.exists():
            raise FileNotFoundError(f"FMC4030 library not found at {path}")

        self._lib = CDLL(str(path))
        self._configure_prototypes()
        self._controller_id: Optional[int] = None

    def _configure_prototypes(self) -> None:
        lib = self._lib

        lib.FMC4030_Open_Device.argtypes = [c_int, c_char_p, c_int]
        lib.FMC4030_Open_Device.restype = c_int

        lib.FMC4030_Close_Device.argtypes = [c_int]
        lib.FMC4030_Close_Device.restype = c_int

        lib.FMC4030_Jog_Single_Axis.argtypes = [
            c_int,
            c_int,
            c_float,
            c_float,
            c_float,
            c_float,
            c_int,
        ]
        lib.FMC4030_Jog_Single_Axis.restype = c_int

        lib.FMC4030_Check_Axis_Is_Stop.argtypes = [c_int, c_int]
        lib.FMC4030_Check_Axis_Is_Stop.restype = c_int

        lib.FMC4030_Home_Single_Axis.argtypes = [
            c_int,
            c_int,
            c_float,
            c_float,
            c_float,
            c_int,
        ]
        lib.FMC4030_Home_Single_Axis.restype = c_int

        lib.FMC4030_Stop_Single_Axis.argtypes = [c_int, c_int, c_int]
        lib.FMC4030_Stop_Single_Axis.restype = c_int

        lib.FMC4030_Get_Axis_Current_Pos.argtypes = [c_int, c_int, POINTER(c_float)]
        lib.FMC4030_Get_Axis_Current_Pos.restype = c_int

        lib.FMC4030_Get_Axis_Current_Speed.argtypes = [c_int, c_int, POINTER(c_float)]
        lib.FMC4030_Get_Axis_Current_Speed.restype = c_int

        lib.FMC4030_Set_Output.argtypes = [c_int, c_int, c_int]
        lib.FMC4030_Set_Output.restype = c_int

        lib.FMC4030_Get_Input.argtypes = [c_int, c_int, POINTER(c_int)]
        lib.FMC4030_Get_Input.restype = c_int

        lib.FMC4030_Write_Data_To_485.argtypes = [c_int, c_char_p, c_int]
        lib.FMC4030_Write_Data_To_485.restype = c_int

        lib.FMC4030_Read_Data_From_485.argtypes = [c_int, c_char_p, POINTER(c_int)]
        lib.FMC4030_Read_Data_From_485.restype = c_int

        self._has_fsc_speed = hasattr(lib, "FMC4030_Set_FSC_Speed")
        if self._has_fsc_speed:
            lib.FMC4030_Set_FSC_Speed.argtypes = [c_int, c_int, c_float]
            lib.FMC4030_Set_FSC_Speed.restype = c_int

        lib.FMC4030_MB01_Operation.argtypes = [
            c_int,
            c_int,
            c_ushort,
            c_char_p,
            POINTER(c_int),
        ]
        lib.FMC4030_MB01_Operation.restype = c_int

        lib.FMC4030_MB03_Operation.argtypes = [
            c_int,
            c_int,
            c_ushort,
            c_int,
            c_char_p,
            POINTER(c_int),
        ]
        lib.FMC4030_MB03_Operation.restype = c_int

        lib.FMC4030_MB05_Operation.argtypes = [
            c_int,
            c_int,
            c_ushort,
            c_ushort,
            c_char_p,
            POINTER(c_int),
        ]
        lib.FMC4030_MB05_Operation.restype = c_int

        lib.FMC4030_MB06_Operation.argtypes = [
            c_int,
            c_int,
            c_ushort,
            c_ushort,
            c_char_p,
            POINTER(c_int),
        ]
        lib.FMC4030_MB06_Operation.restype = c_int

        lib.FMC4030_MB16_Operation.argtypes = [
            c_int,
            c_int,
            c_ushort,
            c_int,
            POINTER(c_ushort),
            c_char_p,
            POINTER(c_int),
        ]
        lib.FMC4030_MB16_Operation.restype = c_int

        lib.FMC4030_Line_2Axis.argtypes = [
            c_int,
            c_uint,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
        ]
        lib.FMC4030_Line_2Axis.restype = c_int

        lib.FMC4030_Line_3Axis.argtypes = [
            c_int,
            c_uint,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
        ]
        lib.FMC4030_Line_3Axis.restype = c_int

        lib.FMC4030_Arc_2Axis.argtypes = [
            c_int,
            c_uint,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
            c_float,
            c_int,
        ]
        lib.FMC4030_Arc_2Axis.restype = c_int

        lib.FMC4030_Pause_Run.argtypes = [c_int, c_uint]
        lib.FMC4030_Pause_Run.restype = c_int

        lib.FMC4030_Resume_Run.argtypes = [c_int, c_uint]
        lib.FMC4030_Resume_Run.restype = c_int

        lib.FMC4030_Stop_Run.argtypes = [c_int]
        lib.FMC4030_Stop_Run.restype = c_int

        lib.FMC4030_Get_Machine_Status.argtypes = [c_int, POINTER(MachineStatus)]
        lib.FMC4030_Get_Machine_Status.restype = c_int

        lib.FMC4030_Get_Device_Para.argtypes = [c_int, POINTER(MachineDeviceParams)]
        lib.FMC4030_Get_Device_Para.restype = c_int

        lib.FMC4030_Set_Device_Para.argtypes = [c_int, POINTER(MachineDeviceParams)]
        lib.FMC4030_Set_Device_Para.restype = c_int

        lib.FMC4030_Get_Version_Info.argtypes = [c_int, POINTER(MachineVersion)]
        lib.FMC4030_Get_Version_Info.restype = c_int

        lib.FMC4030_Download_File.argtypes = [c_int, c_char_p, c_int]
        lib.FMC4030_Download_File.restype = c_int

        lib.FMC4030_Start_Auto_Run.argtypes = [c_int, c_char_p]
        lib.FMC4030_Start_Auto_Run.restype = c_int

        lib.FMC4030_Stop_Auto_Run.argtypes = [c_int]
        lib.FMC4030_Stop_Auto_Run.restype = c_int

        lib.FMC4030_Delete_Script_File.argtypes = [c_int, c_char_p]
        lib.FMC4030_Delete_Script_File.restype = c_int

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def connect(self, config: ControllerConfig) -> None:
        if self._controller_id is not None:
            raise FMC4030Error("Controller is already connected")

        ip_bytes = config.ip.encode("ascii")
        result = self._lib.FMC4030_Open_Device(config.controller_id, ip_bytes, config.port)
        self._check(result, "FMC4030_Open_Device")
        self._controller_id = config.controller_id

    def close(self) -> None:
        if self._controller_id is None:
            return
        result = self._lib.FMC4030_Close_Device(self._controller_id)
        self._check(result, "FMC4030_Close_Device")
        self._controller_id = None

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------
    def jog_single_axis(
        self,
        axis: Axis,
        position_units: float,
        speed_units: float,
        acc_units: float,
        dec_units: float,
        *,
        relative: bool = True,
    ) -> None:
        controller_id = self._require_connection()
        mode = 1 if relative else 2
        result = self._lib.FMC4030_Jog_Single_Axis(
            controller_id,
            int(axis),
            c_float(position_units),
            c_float(speed_units),
            c_float(acc_units),
            c_float(dec_units),
            mode,
        )
        self._check(result, "FMC4030_Jog_Single_Axis")

    def is_axis_stopped(self, axis: Axis) -> bool:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Check_Axis_Is_Stop(controller_id, int(axis))
        if result < 0:
            self._check(result, "FMC4030_Check_Axis_Is_Stop")
        return result == 1

    def home_axis(
        self,
        axis: Axis,
        speed: float,
        acc_dec: float,
        fall_step: float,
        *,
        positive_limit: bool = True,
    ) -> None:
        controller_id = self._require_connection()
        direction = 1 if positive_limit else 2
        result = self._lib.FMC4030_Home_Single_Axis(
            controller_id,
            int(axis),
            c_float(speed),
            c_float(acc_dec),
            c_float(fall_step),
            direction,
        )
        self._check(result, "FMC4030_Home_Single_Axis")

    def stop_axis(self, axis: Axis, mode: int = 2) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Stop_Single_Axis(controller_id, int(axis), mode)
        self._check(result, "FMC4030_Stop_Single_Axis")

    def get_axis_position(self, axis: Axis) -> float:
        controller_id = self._require_connection()
        pos = c_float()
        result = self._lib.FMC4030_Get_Axis_Current_Pos(controller_id, int(axis), byref(pos))
        self._check(result, "FMC4030_Get_Axis_Current_Pos")
        return pos.value

    def get_axis_speed(self, axis: Axis) -> float:
        controller_id = self._require_connection()
        speed = c_float()
        result = self._lib.FMC4030_Get_Axis_Current_Speed(controller_id, int(axis), byref(speed))
        self._check(result, "FMC4030_Get_Axis_Current_Speed")
        return speed.value

    def line_move_2d(
        self,
        axes: Iterable[Axis],
        end_x: float,
        end_y: float,
        speed: float,
        acc: float,
        dec: float,
    ) -> None:
        controller_id = self._require_connection()
        mask = axis_mask(axes)
        result = self._lib.FMC4030_Line_2Axis(
            controller_id,
            mask,
            c_float(end_x),
            c_float(end_y),
            c_float(speed),
            c_float(acc),
            c_float(dec),
        )
        self._check(result, "FMC4030_Line_2Axis")

    def line_move_3d(
        self,
        axes: Iterable[Axis],
        end_x: float,
        end_y: float,
        end_z: float,
        speed: float,
        acc: float,
        dec: float,
    ) -> None:
        controller_id = self._require_connection()
        mask = axis_mask(axes)
        result = self._lib.FMC4030_Line_3Axis(
            controller_id,
            mask,
            c_float(end_x),
            c_float(end_y),
            c_float(end_z),
            c_float(speed),
            c_float(acc),
            c_float(dec),
        )
        self._check(result, "FMC4030_Line_3Axis")

    def arc_move_2d(
        self,
        axes: Iterable[Axis],
        end_x: float,
        end_y: float,
        center_x: float,
        center_y: float,
        radius: float,
        speed: float,
        acc: float,
        dec: float,
        *,
        ccw: bool,
    ) -> None:
        controller_id = self._require_connection()
        mask = axis_mask(axes)
        direction = 1 if ccw else 0
        result = self._lib.FMC4030_Arc_2Axis(
            controller_id,
            mask,
            c_float(end_x),
            c_float(end_y),
            c_float(center_x),
            c_float(center_y),
            c_float(radius),
            c_float(speed),
            c_float(acc),
            c_float(dec),
            direction,
        )
        self._check(result, "FMC4030_Arc_2Axis")

    def pause_run(self, axes_mask: int = 0x07) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Pause_Run(controller_id, axes_mask)
        self._check(result, "FMC4030_Pause_Run")

    def resume_run(self, axes_mask: int = 0x07) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Resume_Run(controller_id, axes_mask)
        self._check(result, "FMC4030_Resume_Run")

    def stop_run(self) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Stop_Run(controller_id)
        self._check(result, "FMC4030_Stop_Run")

    # ------------------------------------------------------------------
    # IO / communication helpers
    # ------------------------------------------------------------------
    def set_output(self, io: int, status: int) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Set_Output(controller_id, io, status)
        self._check(result, "FMC4030_Set_Output")

    def get_input(self, io: int) -> int:
        controller_id = self._require_connection()
        status = c_int()
        result = self._lib.FMC4030_Get_Input(controller_id, io, byref(status))
        self._check(result, "FMC4030_Get_Input")
        return status.value

    def write_data_485(self, payload: bytes) -> None:
        controller_id = self._require_connection()
        buffer = create_string_buffer(payload)
        result = self._lib.FMC4030_Write_Data_To_485(controller_id, buffer, len(payload))
        self._check(result, "FMC4030_Write_Data_To_485")

    def read_data_485(self, max_length: int = 1024) -> bytes:
        controller_id = self._require_connection()
        buffer = create_string_buffer(max_length)
        length = c_int(max_length)
        result = self._lib.FMC4030_Read_Data_From_485(controller_id, buffer, byref(length))
        self._check(result, "FMC4030_Read_Data_From_485")
        return buffer.raw[: length.value]

    def set_fsc_speed(self, slave_id: int, speed: float) -> None:
        controller_id = self._require_connection()
        if not self._has_fsc_speed:
            raise FMC4030Error("FMC4030_Set_FSC_Speed is not available in this library")
        result = self._lib.FMC4030_Set_FSC_Speed(controller_id, slave_id, c_float(speed))
        self._check(result, "FMC4030_Set_FSC_Speed")

    # ------------------------------------------------------------------
    # Modbus helpers
    # ------------------------------------------------------------------
    def mb_read_coils(self, slave_id: int, address: int, count: int) -> bytes:
        controller_id = self._require_connection()
        buffer = create_string_buffer(count)
        length = c_int(count)
        result = self._lib.FMC4030_MB01_Operation(
            controller_id,
            slave_id,
            address,
            buffer,
            byref(length),
        )
        self._check(result, "FMC4030_MB01_Operation")
        return buffer.raw[: length.value]

    def mb_read_holding_registers(self, slave_id: int, address: int, count: int) -> bytes:
        controller_id = self._require_connection()
        byte_length = count * 2
        buffer = create_string_buffer(byte_length)
        length = c_int(byte_length)
        result = self._lib.FMC4030_MB03_Operation(
            controller_id,
            slave_id,
            address,
            count,
            buffer,
            byref(length),
        )
        self._check(result, "FMC4030_MB03_Operation")
        return buffer.raw[: length.value]

    def mb_write_coil(self, slave_id: int, address: int, value: bool) -> bytes:
        controller_id = self._require_connection()
        buffer = create_string_buffer(8)
        length = c_int(8)
        result = self._lib.FMC4030_MB05_Operation(
            controller_id,
            slave_id,
            address,
            0xFF00 if value else 0x0000,
            buffer,
            byref(length),
        )
        self._check(result, "FMC4030_MB05_Operation")
        return buffer.raw[: length.value]

    def mb_write_register(self, slave_id: int, address: int, value: int) -> bytes:
        controller_id = self._require_connection()
        buffer = create_string_buffer(8)
        length = c_int(8)
        result = self._lib.FMC4030_MB06_Operation(
            controller_id,
            slave_id,
            address,
            value & 0xFFFF,
            buffer,
            byref(length),
        )
        self._check(result, "FMC4030_MB06_Operation")
        return buffer.raw[: length.value]

    def mb_write_registers(
        self,
        slave_id: int,
        address: int,
        values: Sequence[int],
    ) -> bytes:
        controller_id = self._require_connection()
        length = len(values)
        array_type = c_ushort * length
        send_buffer = array_type(*[value & 0xFFFF for value in values])
        recv_buffer = create_string_buffer(8)
        recv_len = c_int(8)
        result = self._lib.FMC4030_MB16_Operation(
            controller_id,
            slave_id,
            address,
            length,
            send_buffer,
            recv_buffer,
            byref(recv_len),
        )
        self._check(result, "FMC4030_MB16_Operation")
        return recv_buffer.raw[: recv_len.value]

    # ------------------------------------------------------------------
    # Status / config helpers
    # ------------------------------------------------------------------
    def get_status(self) -> MachineStatus:
        controller_id = self._require_connection()
        status = MachineStatus()
        result = self._lib.FMC4030_Get_Machine_Status(controller_id, byref(status))
        self._check(result, "FMC4030_Get_Machine_Status")
        return status

    def get_device_parameters(self) -> DeviceParameters:
        controller_id = self._require_connection()
        raw = MachineDeviceParams()
        result = self._lib.FMC4030_Get_Device_Para(controller_id, byref(raw))
        self._check(result, "FMC4030_Get_Device_Para")
        return DeviceParameters(
            id=raw.id,
            bound232=raw.bound232,
            bound485=raw.bound485,
            ip=bytes(raw.ip).split(b"\x00", 1)[0].decode("ascii", errors="ignore"),
            port=raw.port,
            div=list(raw.div),
            lead=list(raw.lead),
            soft_limit_max=list(raw.softLimitMax),
            soft_limit_min=list(raw.softLimitMin),
            home_time=list(raw.homeTime),
        )

    def set_device_parameters(self, params: DeviceParameters) -> None:
        controller_id = self._require_connection()
        raw = MachineDeviceParams()
        raw.id = params.id
        raw.bound232 = params.bound232
        raw.bound485 = params.bound485
        max_ip_len = len(raw.ip)
        ip_bytes = params.ip.encode("ascii")[: max_ip_len - 1]
        raw.ip = ip_bytes.ljust(max_ip_len, b"\x00")
        raw.port = params.port
        for idx in range(MAX_AXIS):
            raw.div[idx] = params.div[idx]
            raw.lead[idx] = params.lead[idx]
            raw.softLimitMax[idx] = params.soft_limit_max[idx]
            raw.softLimitMin[idx] = params.soft_limit_min[idx]
            raw.homeTime[idx] = params.home_time[idx]
        result = self._lib.FMC4030_Set_Device_Para(controller_id, byref(raw))
        self._check(result, "FMC4030_Set_Device_Para")

    def get_version_info(self) -> VersionInfo:
        controller_id = self._require_connection()
        raw = MachineVersion()
        result = self._lib.FMC4030_Get_Version_Info(controller_id, byref(raw))
        self._check(result, "FMC4030_Get_Version_Info")
        return VersionInfo(firmware=raw.firmware, library=raw.lib, serial=raw.serialnumber)

    # ------------------------------------------------------------------
    # File / script helpers
    # ------------------------------------------------------------------
    def download_file(self, file_path: Path | str, file_type: int) -> None:
        controller_id = self._require_connection()
        encoded = _encode_str(str(file_path))
        result = self._lib.FMC4030_Download_File(controller_id, encoded, file_type)
        self._check(result, "FMC4030_Download_File")

    def start_auto_run(self, script_name: str) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Start_Auto_Run(controller_id, _encode_str(script_name))
        self._check(result, "FMC4030_Start_Auto_Run")

    def stop_auto_run(self) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Stop_Auto_Run(controller_id)
        self._check(result, "FMC4030_Stop_Auto_Run")

    def delete_script_file(self, script_name: str) -> None:
        controller_id = self._require_connection()
        result = self._lib.FMC4030_Delete_Script_File(controller_id, _encode_str(script_name))
        self._check(result, "FMC4030_Delete_Script_File")

    # ------------------------------------------------------------------
    def _require_connection(self) -> int:
        if self._controller_id is None:
            raise FMC4030Error("Controller is not connected")
        return self._controller_id

    @staticmethod
    def _check(code: int, op_name: str) -> None:
        if code != 0:
            raise FMC4030Error(f"{op_name} failed with error code {code}")

    def __enter__(self) -> "FMC4030Controller":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
