-- Runs Glitcher.fuse outside Fusion, against a stand-in of the parts of the Fuse API it uses
-- (modelled on what GlitcherProbe found in Resolve: float32 rows, bottom row first, reached
-- through a table of row pointers):
--   luajit fusion/test_glitcher_mock.lua fusion/Glitcher.fuse path/to/libglitch_core.so
local fuse, libpath = arg[1], arg[2]
local ffi = require("ffi")

CT_Tool, CHAN_RED = 1, 0
local registered
function FuRegisterClass(name) registered = name end

local W, H = 320, 180
local ImageMT = {}
ImageMT.__index = ImageMT

-- Depth 8 = float32 (scrubbing), Depth 7 = float16 (Resolve's playback), as the debug log showed
local function float_to_half(f)                              -- normal range + zero is enough here
    if f == 0 then return 0 end
    local sign = f < 0 and 0x8000 or 0
    f = math.abs(f)
    local e = math.floor(math.log(f) / math.log(2))
    local m = f / 2 ^ e - 1
    local mi = math.floor(m * 1024 + 0.5)
    if mi == 1024 then mi, e = 0, e + 1 end
    return sign + (e + 15) * 1024 + mi
end
local function half_to_float(hb)
    local sign = hb >= 0x8000 and -1 or 1
    local e, m = math.floor(hb / 1024) % 32, hb % 1024
    if e == 0 then return sign * m * 2 ^ -24 end
    return sign * (1 + m / 1024) * 2 ^ (e - 15)
end

local function new_image(w, h, depth)
    depth = depth or 8
    local cs = depth == 7 and 2 or 4
    local im = setmetatable({ Width = w, Height = h, Depth = depth, cs = cs,
                              buf = ffi.new(cs == 2 and "uint16_t[?]" or "float[?]", w * h * 4),
                              rows = ffi.new("unsigned char*[?]", h) }, ImageMT)
    for y = 0, h - 1 do im.rows[y] = ffi.cast("unsigned char*", im.buf + y * w * 4) end   -- y 0 = bottom
    return im
end
function ImageMT:set(i, v) if self.cs == 2 then self.buf[i] = float_to_half(v) else self.buf[i] = v end end
function ImageMT:get(i) if self.cs == 2 then return half_to_float(self.buf[i]) end return self.buf[i] end
function ImageMT:GetPixel(x, y, p)
    local i = (y * self.Width + x) * 4
    p.R, p.G, p.B, p.A = self:get(i), self:get(i + 1), self:get(i + 2), self:get(i + 3)
end
function Pixel() return { R = 0, G = 0, B = 0, A = 0 } end
function ImageMT:GetChanSize(c) assert(c ~= nil, "invalid arguments"); return self.cs end
function ImageMT:GetScanLine(y) return ffi.cast("unsigned char**", self.rows) end
function Image(attrs) local l = attrs.IMG_Like; return new_image(l.Width, l.Height, l.Depth) end
os.getenv = (function(g) return function(k)
    if k == "TEMP" then return "/tmp" end
    if k == "XDG_CACHE_HOME" then return "/tmp/glitcher_mock_cache" end
    return g(k)
end end)(os.getenv)
function Text(v) return { Value = v } end
function Number(v) return { Value = v } end

-- a moving, textured disc (premultiplied), frame t
local function frame(t, depth)
    local im = new_image(W, H, depth)
    local cx, cy = 90 + 3 * t, 90
    for y = 0, H - 1 do
        for x = 0, W - 1 do
            local dx, dy = (x - cx) / 34, (y - cy) / 70
            local a = math.max(0, math.min(1, (1 - dx * dx - dy * dy) * 8))
            if a > 0 then
                local v = ((x * 7 + y * 3) % 23) / 23
                local i = (y * W + x) * 4
                im:set(i, v * a); im:set(i + 1, 0.5 * a); im:set(i + 2, (1 - v) * a); im:set(i + 3, a)
            end
        end
    end
    return im
end
-- like the log: frames rendered fresh during playback are float16, cached ones float32
local frames = {}
local function at(t) frames[t] = frames[t] or frame(t, (t % 3 == 0) and 8 or 7); return frames[t] end

local values = { Seed = 4, Intensity = 1.0, Tempo = 1.0, Density = 1.0, Bursts = 0.0, FPS = 0,
                 Library = libpath, DebugLog = 1, TintAmount = 0.3, Quality = 0,
                 LayoutBox = 1 }                                -- this frame's box: no shot scan here
local plate_img = nil
local printed = {}
local real_print = print
print = function(...) local t = table.concat({ ... }, " "); printed[#printed + 1] = t; real_print(t) end
for _, e in ipairs({ "Jpeg", "Blocks", "Slices", "Wave", "Sort", "Smear", "Split", "Tint", "Scanlines",
                     "Noise", "Echo", "Stutter", "Mosh" }) do values["Gain" .. e] = 2.0 end
local fetched = {}
local function input(id)
    return {
        GetValue = function(_, req)
            if id == "Input" then return at(req.Time) end
            if id == "Plate" then return plate_img end
            return { Value = values[id] }
        end,
        GetSource = function(_, t)
            if id ~= "Input" and id ~= "Plate" then return { Value = values[id] } end   -- NotifyChanged reads
            fetched[#fetched + 1] = t
            if t < 0 or t > 59 then return nil end
            return at(t)
        end,
        SetSource = function(_, v, t) values[id] = v.Value end,
    }
end
local results = {}
mattes = {}
ids, nests = {}, {}
self = {
    AddInput = function(_, name, id, a)
        if a.LINKID_DataType == "Text" then assert(a.INP_Default == nil, id .. ": use INPS_DefaultText") end
        if a.INP_Default ~= nil then assert(type(a.INP_Default) == "number", id) end
        if a.INPID_InputControl == "FileControl" then assert(a.LINKID_DataType == "Text", id) end
        if a.LBLC_DropDownButton then nests[#nests + 1] = { id = id, n = a.LBLC_NumInputs, first = #ids + 1 } end
        ids[#ids + 1] = id
        if values[id] == nil then
            if a.INPS_DefaultText ~= nil then values[id] = a.INPS_DefaultText else values[id] = a.INP_Default end
        end
        return input(id)
    end,
    AddOutput = function(_, name, id)
        if id == "GlitchMatte" then return { Set = function(_, req, img) mattes[req.Time] = img end } end
        return { Set = function(_, req, img) results[req.Time] = img end }
    end,
    Name = "Glitcher1",
    Comp = { GetPrefs = function() return 25 end,
             -- as Resolve seems to do: the comp's attributes aren't readable while rendering, only
             -- from the UI side (NotifyChanged)
             GetAttrs = function()
                 if not in_notify then error("comp not available while rendering") end
                 return { COMPN_GlobalStart = 0, COMPN_RenderStart = 0, COMPN_RenderEnd = 29 }
             end,
             MapPath = function(_, p) return p end },
}

in_notify = false
dofile(fuse)
assert(registered == "Glitcher")
local fuse_notify = NotifyChanged
function NotifyChanged(...) in_notify = true; fuse_notify(...); in_notify = false end
Create()
local changed, passed = 0, 0
for t = 0, 59 do
    fetched = {}
    Process({ Time = t, IsPreCalc = function() return false end, IsQuick = function() return t % 2 == 0 end })
    local out = results[t]
    assert(out, "no output at " .. t)
    for _, ft in ipairs(fetched) do assert(ft < t and ft >= t - 24 and ft >= 0, "fetched frame " .. ft .. " at " .. t) end
    if out ~= at(t) then
        local diff = false
        for i = 0, W * H * 4 - 1 do
            if out:get(i) ~= at(t):get(i) then diff = true break end
        end
        assert(out.Depth == at(t).Depth, "output format differs from the input's")
        if diff then changed = changed + 1 end
    else
        passed = passed + 1
    end
end
-- a pre-calculation request: no pixels needed, no engine call
local before = #fetched
Process({ Time = 61, IsPreCalc = function() return true end })
assert(results[61] and #fetched == before, "pre-calc did work it didn't need")
print(("mock run OK: %d of 60 frames glitched, %d passed through"):format(changed, passed))
assert(changed > 10, "the engine barely changed anything")
assert(passed == 0, "frames were passed through (engine not used)")

-- every collapsible group holds exactly the inputs that follow it, inside the control list
for _, g in ipairs(nests) do
    assert(g.n and g.n > 0 and g.first + g.n - 1 <= #ids, g.id .. ": bad LBLC_NumInputs")
    for i = g.first, g.first + g.n - 1 do
        for _, h in ipairs(nests) do assert(ids[i] ~= h.id or h == g, "nested group " .. h.id .. " inside " .. g.id) end
    end
end

local function run(n)
    local outs = {}
    for t = 0, n - 1 do
        Process({ Time = t, IsPreCalc = function() return false end })
        outs[t] = results[t]
    end
    return outs
end
local function same(a, b, pred)
    for y = 0, H - 1 do
        for x = 0, W - 1 do
            if not pred or pred(x, y) then
                local i = (y * W + x) * 4
                for c = 0, 3 do
                    if math.abs(a:get(i + c) - b:get(i + c)) > 1e-3 then return false, x, y end
                end
            end
        end
    end
    return true
end
values.DebugLog = 0
local flat = run(30)

-- 3D, from a pose file (turning 0 -> 90 deg) plus a roll slider
local pf = io.open("/tmp/glitcher_mock_pose.csv", "w")
pf:write("# test\nframe,yaw,pitch,roll,scale,confidence\n0,0,10,0,1,1\n30,90,10,0,1,1\n")
pf:close()
values.Mode3D, values.PoseFile, values.Roll = 1, "/tmp/glitcher_mock_pose.csv", 15
values.DebugLog = 1
printed = {}
Process({ Time = 12, IsPreCalc = function() return false end })
local logged = false
for _, l in ipairs(printed) do if l:find("mode=3d%+rolled") and l:find("track=12") then logged = true end end
assert(logged, "debug log has no 3d+rolled line")
values.DebugLog = 0
local d3 = run(30)
local differ = 0
for t = 0, 29 do if not same(flat[t], d3[t]) then differ = differ + 1 end end
assert(differ > 20, "3D mode changed only " .. differ .. " frames")
values.Mode3D, values.PoseFile, values.Roll = 0, "", 0

-- body parts: "Glitch only" the head of a figure standing in the top-right corner, away from
-- the subject: everything left of it must stay exactly the input
local lf = io.open("/tmp/glitcher_mock_lm.csv", "w")
lf:write("size," .. W .. "," .. H .. "\nframe")
for i = 0, 32 do lf:write((",x%d,y%d,v%d"):format(i, i, i)) end
lf:write("\n")
for t = 0, 30, 10 do
    lf:write(t)
    for i = 0, 32 do lf:write((",%d,%d,1"):format(290 + (i % 5) * 4, 20 + math.floor(i / 5) * 6)) end
    lf:write("\n")
end
lf:close()
values.Only_head, values.LandmarksFile = 1, "/tmp/glitcher_mock_lm.csv"
local only = run(30)
for t = 0, 29 do
    local ok, x, y = same(only[t], at(t), function(x) return x < 240 end)
    assert(ok, ("only=head changed (%s,%s) at %d"):format(tostring(x), tostring(y), t))
end
-- without a landmarks file: a warning in the Console, and the frame still renders
values.LandmarksFile = ""
printed = {}
run(2)
local warned = false
for _, l in ipairs(printed) do if l:find("landmarks") then warned = true end end
assert(warned, "no warning for body parts without landmarks")
values.Only_head = 0

-- a plate: opaque result, the clean plate far from the subject (spill 0.05)
plate_img = new_image(W, H, 8)
for y = 0, H - 1 do for x = 0, W - 1 do
    local i = (y * W + x) * 4
    plate_img:set(i, 0.2 + 0.5 * x / W); plate_img:set(i + 1, 0.3); plate_img:set(i + 2, 0.6); plate_img:set(i + 3, 1)
end end
local pl = run(20)
for t = 0, 19 do
    local o = pl[t]
    for _, xy in ipairs({ { 2, 2 }, { W - 3, 2 }, { 2, H - 3 }, { W - 3, H - 3 } }) do
        local i = (xy[2] * W + xy[1]) * 4
        assert(math.abs(o:get(i + 3) - 1) < 1e-3, "plate output not opaque at t=" .. t)
        assert(math.abs(o:get(i) - plate_img:get(i)) < 1e-3, "plate damaged far from the subject at t=" .. t)
    end
end
plate_img = nil
print("mock run OK: 3D, body parts, plate")

-- pose estimation: the Estimate button, frames stored while rendering, a stand-in estimator run
-- in the background, Reload, then the pose in use
os.execute("rm -rf /tmp/glitcher_mock_cache")
local fake = "/tmp/glitcher_mock_estimator"
local ff = io.open(fake, "w")
ff:write([=[#!/usr/bin/env python3
import os, sys
a = sys.argv[1:]
assert a[0] == "--estimate-pose", a
assert "--hint=0:0" in a and "--hint=29:90" in a, a
assert a[a.index("--offset") + 1] == "0" and a[a.index("--variant") + 1] == "lite", a
frames = sorted(os.listdir(a[1]))
assert len(frames) == 30, len(frames)
for i in range(30):
    print(f"landmarks {i + 1}/30", file=sys.stderr, flush=True)
out = a[a.index("-o") + 1]
open(out, "w").write("frame,yaw,pitch,roll\n0,0,10,0\n29,90,10,0\n")
print(f"30 frames -> {out}  (person found in 30)")
]=])
ff:close()
os.execute("chmod +x " .. fake)
values.PoseEstimator, values.YawHints, values.PoseModel = fake, "0:0, 29:90", 2
values.Mode3D, values.PoseFile, values.LandmarksFile = 0, "", ""
NotifyChanged(InBtnEstimate, { Value = 0 }, 0)                -- not a click: nothing
assert(values.PoseRun == 0)
NotifyChanged(InBtnEstimate, { Value = 1 }, 0)
assert(values.PoseRun == 1 and values.Mode3D == 1, "Estimate didn't start a run")
-- the frame range came from the UI side (any control change fills Shot frames in)
assert(values.ShotFrames == "0-29", "Shot frames: " .. tostring(values.ShotFrames))
assert(values.NodeId ~= "", "no node id")
local pdir = values.PoseFolder
assert(pdir:find("^/tmp/glitcher_mock_cache/Glitcher/pose/%d+%-%d+%-%x+%-Glitcher1$"), pdir)
assert(values.PoseFile == pdir .. "/pose.csv" and values.LandmarksFile == pdir .. "/pose_landmarks.csv")
printed = {}
local ready = false
for pass = 1, 200 do
    Process({ Time = pass % 30, IsPreCalc = function() return false end })
    for _, l in ipairs(printed) do if l:find("pose ready") then ready = true end end
    if ready then break end
    os.execute("sleep 0.05")
end
assert(ready, "pose estimation never finished:\n" .. table.concat(printed, "\n"))
local pngs = io.popen("ls " .. pdir .. "/frames | wc -l"):read("*l")
assert(tonumber(pngs) == 30, "stored frames: " .. pngs)
local warned_missing = false
for _, l in ipairs(printed) do if l:find("pose:.*No such file") then warned_missing = true end end
assert(not warned_missing, "warned about the pose file while it was being made")
NotifyChanged(InBtnReload, { Value = 1 }, 0)
assert(values.PoseReloads == 1)
values.DebugLog = 1
printed = {}
Process({ Time = 15, IsPreCalc = function() return false end })
local used = false
for _, l in ipairs(printed) do
    -- frame 15 of a 0 -> 90 deg file over frames 0..29: yaw 46.6
    if l:find("mode=3d") and l:find("warn=0") and l:find("view=yaw 46.6 pitch 10.0") then used = true end
end
assert(used, "the estimated pose isn't used:\n" .. table.concat(printed, "\n"))
-- Show pose draws over the output
local plain = results[15]
values.ShowPose = 1
Process({ Time = 15, IsPreCalc = function() return false end })
assert(not same(plain, results[15]), "Show pose drew nothing")
values.ShowPose = 0
values.DebugLog = 0
-- the same run isn't repeated on another render (e.g. the comp reopened)
printed = {}
Process({ Time = 3, IsPreCalc = function() return false end })
for _, l in ipairs(printed) do assert(not l:find("storing frames"), "run started again") end
print("mock run OK: pose estimation")

-- layout: the whole shot's box (render range 0..29 scanned once, on the first render)
values.Mode3D, values.PoseRun, values.PoseFile, values.LandmarksFile = 0, 0, "", ""
values.LayoutBox, values.DebugLog = 0, 1
fetched, printed = {}, {}
Process({ Time = 5, IsPreCalc = function() return false end })
local seen = {}
for _, f in ipairs(fetched) do seen[f] = (seen[f] or 0) + 1 end
for f = 0, 29 do if f ~= 5 then assert(seen[f], "frame " .. f .. " not scanned") end end
local scanned, box = false, nil
for _, l in ipairs(printed) do
    if l:find("scanned frames 0%-29") then scanned = true end
    box = box or l:match("box=([%d,%-]+)")
end
assert(scanned, "no scan message")
-- the disc moves 3 px a frame from x 90: the shot's box spans frames 0..29, wider than one frame's
local bx0, by0, bx1, by1 = box:match("(%-?%d+),(%-?%d+)%-(%-?%d+),(%-?%d+)")
assert(tonumber(bx1) - tonumber(bx0) > 120, "layout box " .. box .. " isn't the whole shot's")
fetched = {}
Process({ Time = 6, IsPreCalc = function() return false end })
assert(#fetched <= 24, "scanned again")                    -- only history fetches now
NotifyChanged(InBtnRescan, { Value = 1 }, 6)
fetched = {}
Process({ Time = 6, IsPreCalc = function() return false end })
assert(#fetched >= 29, "Rescan didn't scan")

-- hits: Add hit at the playhead, its level in the log, Remove
values.Hits = "40:0.5"
NotifyChanged(InBtnAddHit, { Value = 1 }, 12)
assert(values.Hits == "12, 40:0.5", values.Hits)
printed = {}
Process({ Time = 12, IsPreCalc = function() return false end })
local hitlvl
for _, l in ipairs(printed) do hitlvl = hitlvl or l:match(" hit=([%d.]+)") end
assert(hitlvl == "1.00", "hit level " .. tostring(hitlvl))
NotifyChanged(InBtnDelHit, { Value = 1 }, 13)              -- next to it: removes that one
assert(values.Hits == "40:0.5", values.Hits)
values.Hits = "12:x"
printed = {}
Process({ Time = 12, IsPreCalc = function() return false end })
local complained = false
for _, l in ipairs(printed) do if l:find("hits:") then complained = true end end
assert(complained, "a malformed hit list wasn't reported")
values.Hits, values.DebugLog = "", 0
print("mock run OK: layout box, hits")

-- appear / vanish: buttons edit the lists; hidden before the appear and after the vanish
NotifyChanged(InBtnAppear, { Value = 1 }, 4)
NotifyChanged(InBtnVanish, { Value = 1 }, 20)
values.TransLength = 0.4                                   -- 10 frames at 25 fps
assert(values.Appear == "4" and values.Vanish == "20", values.Appear .. " / " .. values.Vanish)
NotifyChanged(InBtnVanish, { Value = 1 }, 4)               -- a vanish on the appear's frame replaces it
assert(values.Appear == "" and values.Vanish == "4, 20", values.Appear .. " / " .. values.Vanish)
NotifyChanged(InBtnAppear, { Value = 1 }, 4)
assert(values.Appear == "4" and values.Vanish == "20")
NotifyChanged(InBtnDelTrans, { Value = 1 }, 21)            -- removes the vanish next to the playhead
assert(values.Vanish == "", values.Vanish)
NotifyChanged(InBtnVanish, { Value = 1 }, 20)
values.DebugLog = 1
local function state_at(t)
    printed = {}
    Process({ Time = t, IsPreCalc = function() return false end })
    for _, l in ipairs(printed) do local r = l:match(" reveal=(%a+)") if r then return r, results[t] end end
end
local function empty(im)
    for i = 0, W * H * 4 - 1 do if im:get(i) ~= 0 then return false end end
    return true
end
local st, im = state_at(2)
assert(st == "hidden" and empty(im), "before the appear: " .. tostring(st))
st, im = state_at(8)
assert(st == "appearing" and not empty(im), "mid appear: " .. tostring(st))
st = state_at(16)
assert(st == "visible", "between: " .. tostring(st))
st = state_at(24)
assert(st == "vanishing", "mid vanish: " .. tostring(st))
st, im = state_at(35)
assert(st == "hidden" and empty(im), "after the vanish: " .. tostring(st))
-- with a plate: hidden = the clean plate
plate_img = new_image(W, H, 8)
for i = 0, W * H * 4 - 1 do plate_img:set(i, (i % 4 == 3) and 1 or 0.4) end
st, im = state_at(35)
assert(same(im, plate_img), "hidden with a plate isn't the clean plate")
plate_img = nil
values.Appear, values.Vanish, values.DebugLog = "", "", 0
print("mock run OK: appear / vanish")

-- two copies of the node rendering side by side (Resolve runs several script states): one pose
-- run, one estimator process, one shot scan
local env2 = setmetatable({}, { __index = _G })
local chunk = assert(loadfile(fuse))
setfenv(chunk, env2)
chunk()
env2.Create()
os.remove("/tmp/glitcher_mock_runs.txt")
local ff2 = io.open("/tmp/glitcher_mock_estimator2", "w")
ff2:write([=[#!/usr/bin/env python3
import os, sys, time
open("/tmp/glitcher_mock_runs.txt", "a").write("run\n")
time.sleep(0.5)
a = sys.argv[1:]
out = a[a.index("-o") + 1]
open(out, "w").write("frame,yaw\n0,0\n29,90\n")
print(f"30 frames -> {out}  (person found in 30)")
]=])
ff2:close()
os.execute("chmod +x /tmp/glitcher_mock_estimator2")
values.PoseEstimator, values.YawHints, values.PoseFolder = "/tmp/glitcher_mock_estimator2", "", ""
values.LayoutBox = 0
NotifyChanged(InBtnEstimate, { Value = 1 }, 0)
NotifyChanged(InBtnRescan, { Value = 1 }, 0)
local scans = 0
local ready2 = false
for pass = 1, 400 do
    printed = {}
    local P_ = (pass % 2 == 0) and Process or env2.Process
    P_({ Time = pass % 30, IsPreCalc = function() return false end })
    for _, l in ipairs(printed) do
        if l:find("pose ready") then ready2 = true end
        if l:find("layout: scanned") then scans = scans + 1 end
    end
    if ready2 and pass > 20 then break end
    os.execute("sleep 0.02")
end
assert(ready2, "two copies: pose never ready")
local runs = 0
for _ in io.lines("/tmp/glitcher_mock_runs.txt") do runs = runs + 1 end
assert(runs == 1, "the estimator ran " .. runs .. " times")
assert(scans == 1, "the shot was scanned " .. scans .. " times")
-- a range the comp can't give: the Shot frames field decides
values.ShotFrames = "3-12"
values.PoseRun, values.LayoutBox, values.Mode3D = 0, 0, 0
printed = {}
Process({ Time = 5, IsPreCalc = function() return false end })
local r312 = false
for _, l in ipairs(printed) do if l:find("scanned frames 3%-12") then r312 = true end end
assert(r312, "Shot frames 3-12 not used:\n" .. table.concat(printed, "\n"))
print("mock run OK: two copies of the node, Shot frames")

-- the glitch matte (second output): grey, R = G = B = A, black where the frame is the source
values.ShotFrames, values.LayoutBox, values.Mode3D, values.PoseFile = "0-29", 1, 0, ""
values.PoseRun, values.Motion = 0, 0
Process({ Time = 9, IsPreCalc = function() return false end })
local m, o, src = mattes[9], results[9], at(9)
assert(m and m ~= o, "no matte output")
local white = 0
for i = 0, W * H - 1 do
    local v = m:get(4 * i)
    assert(v >= 0 and v <= 1 and m:get(4 * i + 1) == v and m:get(4 * i + 3) == v, "matte not grey")
    local d = 0
    for c = 0, 3 do d = math.max(d, math.abs(o:get(4 * i + c) - src:get(4 * i + c))) end
    if d * 255 < 1 then assert(v == 0, "untouched pixel in the matte") end
    if v > 0.5 then white = white + 1 end
end
assert(white > 20, "matte nearly empty: " .. white)
values.MakeMatte = 0
Process({ Time = 9, IsPreCalc = function() return false end })
for i = 0, W * H * 4 - 1, 97 do assert(mattes[9]:get(i) == 0, "matte off but not blank") end
values.MakeMatte = 1

-- automatic layout: the whole shot in 2D, the follow box in 3D; motion from the scan
values.LayoutBox, values.DebugLog = 0, 1
local function logline(t)
    printed = {}
    Process({ Time = t, IsPreCalc = function() return false end })
    for _, l in ipairs(printed) do if l:find(" layout=") then return l end end
    return ""
end
assert(logline(10):find("layout=shot"), "2D automatic isn't the whole shot")
values.Mode3D = 1
assert(logline(10):find("layout=follow"), "3D automatic isn't the follow box")
values.LayoutBox = 2
assert(logline(10):find("layout=shot"))
values.LayoutBox, values.Mode3D, values.Motion = 1, 0, 0.8
local m1 = tonumber(logline(2):match(" motion=([%-%d.]+)"))
local m2 = tonumber(logline(20):match(" motion=([%-%d.]+)"))
assert(m1 and m2 and m1 >= 0 and m2 >= 0, "no motion value: " .. tostring(m1) .. " " .. tostring(m2))
values.Motion, values.DebugLog = 0, 0
print("mock run OK: glitch matte, follow box, motion")
