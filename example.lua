require 'flandmark'
require 'image'

flandmark.load_model()
local t0 = sys.clock()
im = image.load("1.jpg");
--image.display(im);
res = flandmark.detect(im)
print('detection time = ' .. sys.clock() - t0)
print(res)
flandmark.free()
