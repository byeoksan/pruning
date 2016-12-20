#!/usr/bin/env th

usage = 'Usage: [train|test|prune|onebyone] <options>\n'
if not arg[1] then
    io.stderr:write(usage)
    return
end

command = arg[1]

if command == 'train' or command == 'test' or command == 'prune' or command == 'onebyone' then
    table.remove(arg, 1)
    cmd = require(command)
    cmd.main(arg)
else
    io.stderr:write(usage)
end
