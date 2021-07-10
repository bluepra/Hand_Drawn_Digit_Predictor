import pygame
import torch
from model import Net, model_save_path

# Model
model = torch.load(model_save_path)
 
# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
 
pygame.init()
 
screen_size = (280, 280) # (width, height)
screen = pygame.display.set_mode(screen_size)
 
pygame.display.set_caption("Please draw any digit from 0 to 9")
 
# Loop until the user clicks the close button.
run = True
 
clock = pygame.time.Clock()

# Use the PIL.Image class to reshape and convert image to ndarray and then to tensor
def resize_image(path):
    pass
    
 
while run:
    left_pressed, middle_pressed, right_pressed = pygame.mouse.get_pressed()
    
    for event in pygame.event.get():
        # Quit Handler
        if event.type == pygame.QUIT:
            run = False
        # Left mouse click - draw rect
        elif left_pressed:
            (x,y) = pygame.mouse.get_pos()
            r = pygame.Rect(x,y,28,28)
            pygame.draw.rect(screen, color = WHITE, rect = r)
        # Keyboard key pressed
        elif event.type == pygame.KEYDOWN:
            # Enter 
            if event.key == pygame.K_RETURN:
                image_path = 'user_drawn_image.png'
                pygame.image.save(screen, image_path)
                # image = resize_image(image_path)
                # pred = model(image)
                print('Mr.Digit predicts your drawing to be a')

                # screen.fill(BLACK)
            if event.key == pygame.K_c:
                screen.fill(BLACK)

    # Update screen with newly drawn rects
    pygame.display.flip()
 
    # 60 frames per second
    clock.tick(60)
 
# Close the window and quit.
pygame.quit()