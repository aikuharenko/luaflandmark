require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'inline'
require 'camera'
require 'nnx'
require 'flandmark'
dofile('normalize_image.lua')

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
