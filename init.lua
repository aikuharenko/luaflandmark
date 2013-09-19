require 'torch'
local ffi = require 'ffi'
require 'torchffi'
require 'image'

ffi.cdef[[

	void* load_model();
	void detect(void*, float*, int, int, float* res);
	void free(void*);
	
]]

flandmark = {}

function flandmark.load_model()

	local c = ffi.load("flandmark")
	flandmark.c = c
	flandmark.model = c.load_model()

end

function flandmark.detect(im)

	local width = (#im)[3]
	local height = (#im)[2]
	im0 = image.rgb2y(im)

	img = im:type('torch.FloatTensor'):contiguous()
	local img_c = torch.data(img)

	local res = torch.FloatTensor(16)
	local res_c = torch.data(res)	

	flandmark.c.detect(flandmark.model, img_c, width, height, res_c)

	return res

end

function flandmark.free()
	flandmark.c.free(flandmark.model)
end
