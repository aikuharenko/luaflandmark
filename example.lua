require 'flandmark'
require 'image'

flandmark.load_model()
local t0 = sys.clock()
im = image.lena()--image.load("1.jpg");
--image.display(im);

res = flandmark.find_faces_and_points(im)

print('detection time = ' .. sys.clock() - t0)
print(res)

flandmark.free()
