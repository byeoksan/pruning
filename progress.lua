#!/usr/bin/env th

require 'xlua'

local M = {}

local function getTermLength()
    if sys.uname() == 'windows' then return 80 end
    local tputf = io.popen('tput cols', 'r')
    local w = tonumber(tputf:read('*a'))
    local rc = {tputf:close()}
    if rc[3] == 0 then return w
    else return 80 end 
end

function M.clear()
    local termLength = getTermLength()
    for i = 1, termLength do
        io.write(' ')
    end
    io.write('\r')
end

function M.progress(curr, goal)
    return xlua.progress(curr, goal)
end

return M
