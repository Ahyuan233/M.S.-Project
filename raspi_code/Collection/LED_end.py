import board
import neopixel

# Define the number of pixels in the strip
NUM_LEDS = 1
BRIGHTNESS = 0.5 # 0 to 1

pixel = neopixel.NeoPixel(board.D18, NUM_LEDS, brightness=BRIGHTNESS)

pixel.fill((0,0,0))

# Define the pin connected to the data line of the WS2812 breakout board
#NEOPIXEL_PIN = 15

# Create a neopixel object to control the WS2812 breakout board
#pixels = neopixel.NeoPixel(NEOPIXEL_PIN, NUM_PIXELS, brightness = BRIGHTNESS)

# Set the color of the single pixel to white
#pixels.fill((255, 255, 255))
