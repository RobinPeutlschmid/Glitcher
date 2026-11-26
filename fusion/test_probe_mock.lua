-- Runs GlitcherProbe.fuse outside Fusion against a stand-in of the Fuse API, so syntax and
-- runtime errors show up before it goes into Resolve:
--   luajit fusion/test_probe_mock.lua fusion/GlitcherProbe.fuse path/to/libglitch_core.so
local fuse, libpath = arg[1], arg[2]

CT_Tool = 1
local registered
function FuRegisterClass(name, ct, attrs) registered = name end

-- Images like Resolve's: float32 RGBA rows, stored bottom-up (row 0 = bottom, at the highest
-- address), reached through GetScanLine - the layout the probe has to work out by itself.
local ffi = require("ffi")
local Image = {}
Image.__index = Image
local function rowp(self, y) return self.buf + (self.Height - 1 - y) * self.Width * 4 end
function Image.new(w, h)
    local im = setmetatable({ Width = w, Height = h, Depth = 7, buf = ffi.new("float[?]", w * h * 4 + 64) }, Image)
    for y = 0, h - 1 do
        for x = 0, w - 1 do
            local inside = math.abs(x - w / 2) < w / 8 and math.abs(y - h / 2) < h / 3
            local r = rowp(im, y)
            r[4 * x], r[4 * x + 1], r[4 * x + 2], r[4 * x + 3] = inside and 0.5 or 0, inside and 0.4 or 0, inside and 0.3 or 0, inside and 1 or 0
        end
    end
    return im
end
function Image:GetChanSize(c) assert(c ~= nil, "invalid arguments"); return 4 end
function Image:GetData() return nil end
local slot = ffi.new("unsigned char*[1]")                -- one shared slot, like Resolve's
function Image:GetScanLine(y)
    slot[0] = ffi.cast("unsigned char*", rowp(self, y))
    return ffi.cast("unsigned char**", slot)
end
function Image:GetPixel(x, y, p)
    local r = rowp(self, y)
    p.R, p.G, p.B, p.A = r[4 * x], r[4 * x + 1], r[4 * x + 2], r[4 * x + 3]
end
function Image:CopyOf()
    local c = setmetatable({ Width = self.Width, Height = self.Height, Depth = self.Depth,
                             buf = ffi.new("float[?]", self.Width * self.Height * 4 + 64) }, Image)
    ffi.copy(c.buf, self.buf, self.Width * self.Height * 16)
    return c
end
function Pixel() return { R = 0, G = 0, B = 0, A = 0 } end

local function input(value)
    return {
        GetValue = function(_, req) return type(value) == "table" and value or { Value = value } end,
        GetSource = function(_, t) return Image.new(640, 360) end,
    }
end

local outputs = {}
local self_ = { AddInput = function(_, name, id, a)
                    -- like Fusion: a Text input takes INPS_DefaultText; INP_Default is for numbers
                    if a.LINKID_DataType == "Text" then
                        assert(a.INP_Default == nil, id .. ": Text inputs need INPS_DefaultText, not INP_Default")
                    end
                    if a.INP_Default ~= nil then
                        assert(type(a.INP_Default) == "number", id .. ": INP_Default must be a number")
                    end
                    if id == "Library" then return input(libpath) end
                    if id == "Lag" then return input(12) end
                    the_input = Image.new(640, 360)
                    return input(the_input)
                end,
                AddOutput = function(_, name, id, a) return { Set = function(_, req, img) outputs[#outputs + 1] = img end } end }
self = self_

dofile(fuse)
assert(registered == "GlitcherProbe", "not registered")
Create()
Process({ Time = 30 })
Process({ Time = 31 })                                    -- library stays loaded
assert(#outputs == 2, "no output set")
if libpath ~= "" then
    -- the engine wrote into the output: a ghost right of the subject, the input untouched
    local o, w, h = outputs[2], outputs[2].Width, outputs[2].Height
    local row = ffi.cast("float*", o:GetScanLine(math.floor(h / 2))[0])
    local x = math.floor(w / 2 + w / 8 + w * 0.01)
    assert(row[4 * x + 3] > 0, "no echo in the output")
    assert(o.buf ~= the_input.buf, "output is the input")
end
print("mock run OK")
