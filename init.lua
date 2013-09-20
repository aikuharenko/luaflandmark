require 'torch'
local ffi = require 'ffi'
require 'torchffi'
require 'image'

ffi.cdef[[

	void* load_model();
	void detect(void*, float*, int, int, float* res);
	void free(void*);
	void find_faces_and_points(void* vmodel, float* f_im, int width, int height, float* res);
	
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

	local res = torch.FloatTensor(1000)
	local res_c = torch.data(res)	

	flandmark.c.detect(flandmark.model, img_c, width, height, res_c)

	return res

end

function flandmark.find_faces_and_points(im)

	local width = (#im)[3]
	local height = (#im)[2]
	local im0 = image.rgb2y(im)

	local img = im0:type('torch.FloatTensor'):contiguous()
	local img_c = torch.data(img)

	local res = torch.FloatTensor(1000)
	local res_c = torch.data(res)	

	flandmark.c.find_faces_and_points(flandmark.model, img_c, width, height, res_c)

	local detections = {}
	for i = 1, res[1] do
		
		local elem = {}
		elem.bbox = res[{{2 + 20 * (i - 1), 5 + 20 * (i - 1)}}]
		
		local left_eye = {}
		left_eye.x2 = res[20 * (i - 1) + 8];
		left_eye.y2 = res[20 * (i - 1) + 9];
		left_eye.x1 = res[20 * (i - 1) + 16];
		left_eye.y1 = res[20 * (i - 1) + 17];
		elem.left_eye = left_eye
		
		local right_eye = {}
		right_eye.x1 = res[20 * (i - 1) + 10];
		right_eye.y1 = res[20 * (i - 1) + 11];
		right_eye.x2 = res[20 * (i - 1) + 18];
		right_eye.y2 = res[20 * (i - 1) + 19];
		elem.right_eye = right_eye

		local nose = {}
		nose.x = res[20 * (i - 1) + 6];
		nose.y = res[20 * (i - 1) + 7];
		elem.nose = nose
		
		local lips = {}
		lips.x1 = res[20 * (i - 1) + 12];
		lips.y1 = res[20 * (i - 1) + 13];
		lips.x2 = res[20 * (i - 1) + 14];
		lips.y2 = res[20 * (i - 1) + 15];
		elem.lips = lips		

		detections[i] = elem 
		
	end
	
	return detections

end

--[[function flandmark.find_faces(im)

	local width = (#im)[3]
	local height = (#im)[2]
	im0 = image.rgb2y(im)

	img = im:type('torch.FloatTensor'):contiguous()
	local img_c = torch.data(img)
	
	flandmark.c.find_faces(img_c, width, height)
	
end--]]


function flandmark.free()
	flandmark.c.free(flandmark.model)
end
