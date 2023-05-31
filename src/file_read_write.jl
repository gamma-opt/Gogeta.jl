# writes data (variable) to a file
function write2file(file_name, data)
    file = open(file_name, "w")
    serialize(file, data)
    close(file)
end

# reads data (variable) from a file
function readfile(file_name)
    file = open(file_name)
    data = deserialize(file)
    close(file)
    return data
end