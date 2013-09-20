require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'inline'
require 'camera'
require 'nnx'
require 'flandmark'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='camera index: /dev/videoIDX', default=0}
op:option{'-n', '--network', action='store', dest='network', 
          help='path to existing [trained] network',
          default='face.net.ascii'}
opt,args = op:parse()
opt.eye_dist = 50
opt.dx = 64
opt.dtop = 40
opt.dbottom = 88

torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(4)

-- setup camera
camera = image.Camera(opt.camidx)


-- setup GUI (external UI file)
if not win or not widget then 
   widget = qtuiloader.load('g.ui')
   win = qt.QtLuaPainter(widget.frame) 
end

if not win2 or not widget2 then 
   widget2 = qtuiloader.load('g2.ui')
   win2 = qt.QtLuaPainter(widget2.frame) 
end

flandmark.load_model()

-- profiler
p = xlua.Profiler()

-- process function
function process()
   -- (1) grab frame
   frame = camera:forward()

   -- (2) transform it into Y space
   --frameY = image.rgb2y(frame)
 
   -- (7) clean up results
   detections = {}
   detections = flandmark.find_faces_and_points(frame)

end

-- display function
function display()
   win:gbegin()
   win:showpage()
   -- (1) display input image + pyramid
   image.display{image=frame, win=win, saturation=false, min=0, max=1}

   -- (2) overlay bounding boxes for each detection
   for i,detect in ipairs(detections) do
      win:setcolor(0,1,0)
      win:rectangle(detect.bbox[1], detect.bbox[2], detect.bbox[3], detect.bbox[4])
      win:stroke()
      win:setfont(qt.QFont{serif=false,italic=false,size=16})
      win:moveto(detect.bbox[1], detect.bbox[2] - 1)
      win:show('face')
      
      win:rectangle(detect.left_eye.x1 - 2, detect.left_eye.y1 - 2, 4, 4);
      win:rectangle(detect.left_eye.x2 - 2, detect.left_eye.y2 - 2, 4, 4);
      win:rectangle(detect.right_eye.x1 - 2, detect.right_eye.y1 - 2, 4, 4);
      win:rectangle(detect.right_eye.x2 - 2, detect.right_eye.y2 - 2, 4, 4);
      win:rectangle(detect.lips.x1 - 2, detect.lips.y1 - 2, 4, 4);
      win:rectangle(detect.lips.x2 - 2, detect.lips.y2 - 2, 4, 4);
      win:rectangle(detect.nose.x - 2, detect.nose.y - 2, 4, 4);
      
 
   end
   win:gend()
end

function normalize_image(im, points)

	local x1 = points.bbox[1]
	local x2 = points.bbox[1] + points.bbox[3] - 1
	local y1 = points.bbox[2]
	local y2 = points.bbox[2] + points.bbox[4] - 1
	
	local eye_x1 = (points.left_eye.x1 + points.left_eye.x2) / 2
	local eye_y1 = (points.left_eye.y1 + points.left_eye.y2) / 2
	local eye_x2 = (points.right_eye.x1 + points.right_eye.x2) / 2
	local eye_y2 = (points.right_eye.y1 + points.right_eye.y2) / 2
	
	local eye_dist = math.sqrt( (eye_x1 - eye_x2) ^ 2 + (eye_y1 - eye_y2) ^ 2)
	local sin_a = (eye_y2 - eye_y1) / eye_dist
	local cos_a = (eye_x2 - eye_x1) / eye_dist
	
	local alpha = torch.asin(sin_a)
	local sx = (#im)[3]
	local sy = (#im)[2]
	
	local im_x1 = math.max(1, x1 - eye_dist)
	local im_y1 = math.max(1, y1 - eye_dist)
	local im_x2 = math.min(sx, x2 + eye_dist)
	local im_y2 = math.min(sy, y2 + eye_dist)
	
	eye_x1 = eye_x1 - im_x1
	eye_y1 = eye_y1 - im_y1
	eye_x2 = eye_x2 - im_x1
	eye_y2 = eye_y2 - im_y1
	
	local im2 = im[{{},{im_y1, im_y2}, {im_x1, im_x2}}]:clone()
	local sx = (#im2)[3]
	local sy = (#im2)[2]
	local cx = sx / 2
	local cy = sy / 2
	
	--rotate
	im2 = image.rotate(im2, alpha)
	
	--scale
	local sc = opt.eye_dist / eye_dist
	im2 = image.scale(im2, sx * sc, sy * sc, 'bilinear')
	
	--new eyes coordinates
	im2_left_eye_x = ((eye_x1 - cx) * cos_a + (eye_y1 - cy) * sin_a + cx) * sc
	im2_left_eye_y = (-(eye_x1 - cx) * sin_a + (eye_y1 - cy) * cos_a + cy) * sc
	im2_right_eye_x = ((eye_x2 - cx) * cos_a + (eye_y2 - cy) * sin_a + cx) * sc
	im2_right_eye_y = (-(eye_x2 - cx) * sin_a + (eye_y2 - cy) * cos_a + cy) * sc
	
	im2_bbox = {x1, y1, x2 - x1 + 1, y2 - y1 + 1}
	
	--crop
	local eye_cx = math.floor((im2_left_eye_x + im2_right_eye_x) / 2)
	local eye_cy = math.floor(im2_left_eye_y)
	
	local xn1 = eye_cx - opt.dx + 1
	local xn2 = eye_cx + opt.dx
	local yn1 = eye_cy - opt.dtop + 1
	local yn2 = eye_cy + opt.dbottom
	--print(xn1, yn1, xn2, yn2, sx, sy)
	if (xn1 < 0) then
		local add_x = 1 - xn1
		im2 = image.translate(im2, add_x, 0)
		xn1 = 1
		xn2 = xn2 + add_x
		
	end
	
	if (yn1 < 0) then
		local add_y = 1 - yn1
		im2 = image.translate(im2, 0, add_y)
		yn1 = 1
		yn2 = yn2 + add_y
	end

	if (xn2 > (#im2)[3]) then
		local add_x = xn2 - (#im2)[3]
		im2 = image.translate(im2, -add_x, 0)
		xn2 = (#im2)[3]
		xn1 = xn1 - add_x
	end
	
	if (yn2 > (#im2)[2]) then
		local add_y = yn2 - (#im2)[2]
		im2 = image.translate(im2, 0, -add_y)
		yn2 = (#im2)[2]
		yn1 = yn1 - add_y
	end
	--print(xn1, yn1, xn2, yn2)
	return im2[{{}, {yn1, yn2}, {xn1, xn2}}]
	
end

function display2()
	win2:gbegin()
	win2:showpage()
	
	local x0 = 1
	local y0 = 1
	local sx = 640
	local sy = 480
	local res_im = torch.Tensor(3, sy, sx):fill(1)
	
	for i,detect in ipairs(detections) do
		
		if (detections[i].left_eye.x1 > 0) then
		
			local im2 = normalize_image(frame, detections[i])
			local sx0 = (#im2)[3]
			local sy0 = (#im2)[2]
			
			
			if (y0 + sy0 < sy) then
				res_im[{{}, {y0, y0 + sy0 - 1}, {x0, x0 + sx0 - 1}}] = im2
				x0 = x0 + 2 * opt.dx + 10

				if (x0 + sx0 > sx) then
					x0 = 1
					y0 = y0 + opt.dtop + opt.dbottom + 10
				end	
				
			end	
			
			--win2:setcolor(0,1,0)
			--win2:rectangle(im2_bbox[1], im2_bbox[2], im2_bbox[3], im2_bbox[4])
			--win2:rectangle(im2_left_eye_x - 2, im2_left_eye_y - 2, 4, 4);
			--win2:rectangle(im2_right_eye_x - 2, im2_right_eye_y - 2, 4, 4);
			
		end
		
	end

	image.display{image=res_im, win=win2, saturation=false, min=0, max=1}
	win2:stroke()
		
	win2:gend()

end

-- setup gui
timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              p:start('full loop','fps')
              p:start('prediction','fps')
              process()
              p:lap('prediction')
              p:start('display','fps')
              display()
              display2()
              p:lap('display')
              timer:start()
              p:lap('full loop')
              p:printAll()
           end)
widget.windowTitle = 'Face Detector'
widget:show()
widget2:show()
timer:start()
