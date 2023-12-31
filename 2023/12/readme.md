# 2023, Decemeber.

This was my first challenge for Pythonista.

Written: Dec 2 thru Dec 4

Time: ~9 hours of thought process and development

## Intuition
My initial thoughts were to find each sensor, then associate a radius 
with that sensor (ie, Manhattan  distance to the nearest tag). Finally, 
I would redraw the map and, for each cell on the map, checking if the cell 
was within the radius of any sensor.

### Implementation:
1. Convert string array into np array of 0, 1, and 2s
   (0 = empty, 1 = sensor, 2 = tag)
2. Store the indicies of sesnors
3. For every sesnsor, do a simple BFS using a deque object and store
   them (radii={sesnor_location: radius})
4. Create the new map:
   1. Scan through a new array of 1s in the shape of the input grid
   2. For each index, if it is within the radius of any sensor, mark
      it as known (0). If not, mark it as unknown (1)


## Revisions
After implementing this, however, I realized that it was not very efficient
(at least, not in theory) because I iterate over the array an unnecessary
amount of times.

My next, and current, implementation was more efficent by doing more in 
one loop. In the Breadth-First-Search, I scan in waves around the sesnor. 
Instead of scanning once to find the radii, then again to draw, I simply
scan once and generate the cells which do not need to be searched, then
combine them to create a joint map.

### Implementation:
1. (see intuition approach)
2. (see intuition approach)
3. For every sensor, do a simple BFS. This BFS does:
   1. Normal BFS using a deque to find the nearest tag by the sensor
   2. Once a tag was found, begin to mark cells that don't need to be searched. 
   3. Create a simple diamond shape centered around the sensor (since that is the boundary shape 
      is created when using Manhattan distance instead of Euclidian).
   4. For each cell within the diamond, mark it on a new map in the shape of the original grid 
       of 0, 1, 2s.
4. For each sesnor's map, I use `AND` logic to combine a base map and the new map - meaning, 
  cells that have been discovered by ANY sensor will be marked as known on the final map. Only
  cells that are not knowm by any sensor, ever, will be marked as unknown.  (0 = known, 1 = unknown)
5. Convert this map into a 2D Python list and return it.

## Thoughts
This was a fun problem to solve. I recently learned about BFS and how to implement it, and it 
was very enjoyable to solve this as efficiently as possible. I now feel more confident in my abilities
to use BFS to solve actual problems.
