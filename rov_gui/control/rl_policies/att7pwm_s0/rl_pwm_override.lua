-- rl_pwm_override.lua — ArduSub (Navigator, Sub-4.5) vehicle-side gate for the RL "pwm" policy (2026-09-27).
--
-- STATUS: WRITTEN FROM THE Sub-4.5 SCRIPTING BINDINGS AND THE mavlink_message_t LAYOUT, NOT YET RUN ON THE VEHICLE [예측].
-- Bench-test with the propellers removed before any water use (RL_controller/README.md §10.3). Two independent reviewers
-- (2026-09-27) checked the header/field offsets below against pymavlink's mavlink_types.h and ArduPilot's
-- modules/MAVLink/mavlink_msgs.lua; the first draft had them wrong.
--
-- Why a script and not SERVOn_FUNCTION = RCIN passthrough: passthrough outputs ignore ARM/DISARM and every failsafe, and on
-- this receiver-less ROV a dead topside bridge would latch the last pulse forever. Here SERVO1..8 STAY Motor1..8 and the
-- script FORCES each output with SRV_Channels:set_output_pwm_chan_timeout(chan, pwm, TIMEOUT_MS): while the topside sends
-- fresh RC_CHANNELS_OVERRIDE frames (channels 9..16 = motors 1..8) AND the vehicle is armed AND in MANUAL, the policy's
-- pulses go out; when frames stop the script forces neutral once and lets the override expire, so AP_Motors resumes (its
-- own output is the neutral MANUAL_CONTROL mixer = 1500 us, or 1500 us when disarmed). DISARM, E-stop, leak, battery,
-- pilot-input and GCS failsafes therefore all still stop the thrusters. Worst-case hold of a stale pulse after a topside
-- crash: STALE_MS + TIMEOUT_MS = 200 ms.
--
-- Vehicle parameters (all read back by the bridge before it engages):
--   SCR_ENABLE 1 (reboot), SCR_HEAP_SIZE >= 100000, SCR_USER1 = 1 (arms THIS script; 0 disables it without removing it)
--   SERVO1..8_FUNCTION = 33..40 (Motor1..8, unchanged), MOT_1..8_DIRECTION as in the exported obs_spec.json
--   RC_OPTIONS bit1 (ignore MAVLink overrides) = 0, RC_OVERRIDE_TIME > 0, RC9..16_OPTION = 0, no SERVOn_FUNCTION = RCIN9..16
--   SYSID_MYGCS = bridge source_system (frames from any other sysid are DROPPED here too: AP_Scripting hands scripts every
--   registered message before the GCS sysid filter), SYSID_THISMAV = target_system the bridge addresses.
-- Topside: deploy/ardusub_bridge.py send_pwm -> RC_CHANNELS_OVERRIDE(chan9..16 = pulses, others 65535) at 20 Hz plus a
-- neutral MANUAL_CONTROL (keeps the mixer inputs neutral, so the release fallback is 1500 us). The script answers with
-- NAMED_VALUE_FLOAT "RLPWM" at 2 Hz (1 = forcing, 0 = idle) and the bridge refuses to keep sending unless it sees 1.

local MSG_RC_CHANNELS_OVERRIDE = 70
local MANUAL_MODE = 19
local TIMEOUT_MS = 100          -- override lifetime per frame; the bridge sends at 20 Hz (50 ms)
local STALE_MS = 100            -- no fresh frame for this long -> force neutral once, then let the override expire
local PWM_MIN, PWM_MAX = 1100, 1900
local LOOP_MS = 20

local user_enable = Parameter()
assert(user_enable:init("SCR_USER1"), "SCR_USER1 missing")
local sysid_mygcs = Parameter()
assert(sysid_mygcs:init("SYSID_MYGCS"), "SYSID_MYGCS missing")
local sysid_thismav = Parameter()
assert(sysid_thismav:init("SYSID_THISMAV"), "SYSID_THISMAV missing")

-- (rx queue depth, number of registered msgids) per Sub-4.5 lua_bindings.cpp; docs.lua lists the two the other way round.
-- A full queue drops the NEWEST frame, so keep the depth small and drain it every tick.
mavlink:init(4, 1)
mavlink:register_rx_msgid(MSG_RC_CHANNELS_OVERRIDE)

local last_rx_ms = 0
local pwm = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500}
local engaged = false
local last_hb_ms = 0

-- receive_chan() returns the raw mavlink_message_t struct: [1-2] checksum, [3] magic, [4] len, [5] incompat_flags,
-- [6] compat_flags, [7] seq, [8] sysid, [9] compid, [10-12] msgid (24-bit LE), [13 ..] payload of `len` bytes
-- (pymavlink include_v2.0/mavlink_types.h; ArduPilot modules/MAVLink/mavlink_msgs.lua decode_header reads from marker 3).
-- RC_CHANNELS_OVERRIDE wire order (fields sorted by size, extensions appended): chan1..chan8 (uint16 x 8 = bytes 1..16),
-- target_system (17), target_component (18), chan9..chan18 (uint16, bytes 19..38). MAVLink 2 trims trailing zero bytes,
-- so a channel beyond `len` reads as 0 = ignore. On channels 9..16: 0 and 65535 = leave, 65534 = release (Sub-4.5).
local function handle(msg)
    local len = string.byte(msg, 4) or 0
    local sysid = string.byte(msg, 8) or -1
    local msgid = string.unpack("<I3", msg, 10)
    if msgid ~= MSG_RC_CHANNELS_OVERRIDE then return false end
    if sysid ~= sysid_mygcs:get() then return false end             -- not our GCS: never drive motors from a stray sender
    local payload = string.sub(msg, 13, 12 + len)
    local target_system = (#payload >= 17) and string.byte(payload, 17) or 0
    if target_system ~= sysid_thismav:get() then return false end
    local any = false
    for m = 1, 8 do
        local off = 19 + 2 * (m - 1)                                -- chan(8+m)
        local v = 0
        if off + 1 <= #payload then v = string.unpack("<I2", payload, off) end
        if v ~= 0 and v ~= 65535 and v ~= 65534 then
            if v < PWM_MIN then v = PWM_MIN end
            if v > PWM_MAX then v = PWM_MAX end
            pwm[m] = v
            any = true
        end
    end
    return any
end

local function force(values)
    for m = 1, 8 do
        SRV_Channels:set_output_pwm_chan_timeout(m - 1, values[m], TIMEOUT_MS)   -- chan is 0-based (output 1 = 0)
    end
end

local NEUTRAL = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500}

local function update()
    local now = millis():toint()
    while true do                                                  -- drain the queue; keep the newest frame
        local msg, chan, ts = mavlink:receive_chan()
        if not msg then break end
        if handle(msg) then last_rx_ms = now end
    end
    local fresh = (now - last_rx_ms) < STALE_MS
    local ok = (user_enable:get() == 1) and arming:is_armed() and (vehicle:get_mode() == MANUAL_MODE) and fresh
    if ok then
        force(pwm)
    elseif engaged then
        force(NEUTRAL)                                             -- one neutral frame, then the override expires on its own
    end
    if ok ~= engaged then
        engaged = ok
        gcs:send_text(6, engaged and "RLPWM: engaged (policy pulses forced on motors 1..8)" or "RLPWM: released (AP_Motors output)")
    end
    if now - last_hb_ms >= 500 then
        last_hb_ms = now
        gcs:send_named_float("RLPWM", engaged and 1 or 0)
    end
    return update, LOOP_MS
end

gcs:send_text(6, "rl_pwm_override.lua loaded (SCR_USER1=1 to enable)")
return update, 1000
