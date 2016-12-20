#!/usr/bin/env th

usage = 'Usage: [train|test|prune|probe-impact] <options>\n'
if not arg[1] then
    io.stderr:write(usage)
    return
end

command = arg[1]

if command == 'train' or command == 'test' or command == 'prune' or command == 'probe-impact' then
    table.remove(arg, 1)
    cmd = require(command:gsub('-', '_'))
    cmd.main(arg)
else
    io.stderr:write(usage)
end
