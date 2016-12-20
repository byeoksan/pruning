#!/usr/bin/env th

local M = {}

function M.csvToList(csv)
    if torch.type(csv) ~= 'string' then
        return List{}
    end

    csv = string.gsub(csv, ' ', '')
    list = List(string.split(csv, ',')):map(tonumber)

    if list:contains(false) then
        return List{}
    end

    return list
end

return M
