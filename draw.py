import pygame
import torch
from model import FC_Net, Net, model_save_path
from PIL import Image
import numpy as np

# Use the PIL.Image class to reshape and convert image to ndarray and then to tensor
def resize_image(path):
    # Open image 
    image = Image.open(path).convert('L')

    # Resize to (28,28)
    new_width = 28
    new_size = (new_width, new_width)
    image = image.resize(new_size)
    image.save('./images/resized_gray.png')

    arr = np.array(image.getdata())
    arr = np.reshape(arr, new_size) / 255
    # print(arr)

    t = torch.as_tensor(arr)
    t = torch.reshape(t, (1, 1, new_width, new_width))

    return t

def main():
    # Model
    print('Using model:', model_save_path)
    model = torch.load(model_save_path)

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
     
    pygame.init()
     
    screen_size = (280, 280) # (width, height)
    screen = pygame.display.set_mode(screen_size)
     
    pygame.display.set_caption("Draw any digit 0 to 9")
     
    # Loop until the user clicks the close button.
    run = True
     
    clock = pygame.time.Clock()

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
                    image_path = './images/user_drawn_image.png'
                    pygame.image.save(screen, image_path)
                    image = resize_image(image_path)
                    # print(image)
                    pred = model(image.float())
                    # print(pred)
                    pred = round(pred.item())
                    print('Your drawing is a', pred)

                    # screen.fill(BLACK)
                if event.key == pygame.K_c:
                    screen.fill(BLACK)

        # Update screen with newly drawn rects
        pygame.display.flip()
     
        # 60 frames per second
        clock.tick(60)
     
    # Close the window and quit.
    pygame.quit()

if __name__ == '__main__':
    main()