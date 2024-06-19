def convert_chunk(input_string):
    # Split the input string into lines
    lines = input_string.splitlines()
    
    # Initialize an empty 2D list
    array_2d = []
    
    # Iterate through each line
    for line in lines:
        # Create a list to store characters of current line
        row = []
        
        # Iterate through each character in the line
        for char in line:
            row.append(char)
        
        # Add the row to the 2D array
        array_2d.append(row)
    
    return array_2d

def print_surrounding_grid(array_2d, size):
    # Get dimensions of the 2D array
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows > 0 else 0
    
    # Iterate through each element in the 2D array
    for i in range(rows):
        for j in range(cols):
            print("{:02}".format(i) + "{:02}".format(j), end="")
            for x in range(size - 1,-1,-1):
                for y in range(size - 1,-1,-1):
                    if x == 0 and y == 0:
                        break
                    elif i - x < 0 or j - y < 0:
                        print(' ', end="")
                    else:
                        print(array_2d[i - x][j - y], end="")
            print("}" + array_2d[i][j])



# main
file = "raw_data.txt"
size = 3


# Open the file
with open(file, 'r') as f:
    # Seek to the start position
    f.seek(0)
    line = f.readline()
    while line:
        # Read until you hit the <start> marker
        chunk = ''
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() == '<start>':
                break
            chunk += line

        # Read until you hit the <end> marker
        while True:
            line = f.readline()
            if not line or line.strip() == '<end>':
                break
            chunk += line
        print_surrounding_grid(convert_chunk(chunk), size)
     

# Close the file
f.close()

