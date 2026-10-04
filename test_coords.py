import cv2, numpy as np
image = cv2.imread('uploads/3334c1ba-f524-4f7c-9f34-d4d19e6b6c34.jpg')
std_height=1200
ratio = std_height / image.shape[0]
std_width = int(image.shape[1] * ratio)
resized = cv2.resize(image, (std_width, std_height))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(thresh).reshape(-1, 2)
x_coords = coords[:, 0]
y_coords = coords[:, 1]
page_left_x = int(np.min(x_coords))
page_right_x = int(np.max(x_coords))
page_width = page_right_x - page_left_x
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_bubbles=[]
for c in cnts:
  x, y, w, h = cv2.boundingRect(c)
  ar = w / float(h)
  if 10 <= w <= 50 and 10 <= h <= 50 and 0.7 <= ar <= 1.3:
      mask = np.zeros(thresh.shape, dtype='uint8')
      cv2.drawContours(mask, [c], -1, 255, -1)
      if cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)) > (w*h*0.60):
          filled_bubbles.append({'x': x+w//2, 'y': y+h//2, 'w': w, 'h': h})

col_width = page_width / 3.0
print('page_width', page_width, 'col_width', col_width)
for i in range(3):
  min_x = page_left_x + i * col_width
  max_x = page_left_x + (i + 1) * col_width
  bubs = [b for b in filled_bubbles if min_x < b['x'] < max_x]
  bubs.sort(key=lambda b: b['y'])
  print(f'Col {i}:')
  for b in bubs:
     local_x = b['x'] - min_x
     zone_start = 0.22 * col_width
     zone_end = 0.95 * col_width
     slot = (local_x - zone_start) / ((zone_end - zone_start) / 4.0)
     idx = int(max(0, min(3, slot)))
     print(f"  y={b['y']} x={b['x']} local_x={local_x:.1f} slot={slot:.2f} idx={idx}")
