import io
import base64
import numpy as np
from PIL import Image

def get_scroll_position(page):
    return page.evaluate("""() => {
        const scrollTop = window.scrollY;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const remainingPixels = documentHeight - (scrollTop + windowHeight);

        return {
            'scrollTop': scrollTop,
            'windowHeight': windowHeight,
            'documentHeight': documentHeight,
            'remainingPixels': remainingPixels
        };
    }""")
    
def image_to_jpg_base64_url(
    image: np.ndarray | Image.Image, add_data_prefix: bool = False
):
    """Convert a numpy array to a base64 encoded jpeg image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    width, height = image.size
    # logger.info(f'Width: {width}, Height: {height}')
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG', quality=10)

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return (
        f'data:image/jpeg;base64,{image_base64}'
        if add_data_prefix
        else f'{image_base64}'
    )
    
def get_serializable_obs(env, obs):
    scroll_position = get_scroll_position(env.page)
    obs['scroll_position'] = scroll_position
    # make observation serializable
    obs['screenshot'] = image_to_jpg_base64_url(obs['screenshot'])
    obs['active_page_index'] = obs['active_page_index'].item()
    obs['elapsed_time'] = obs['elapsed_time'].item()
    return obs


# Define a custom exception for timeout
class TimeoutException(Exception): ...

# Function to handle the alarm signal
def timeout_handler(signum, frame):
    raise TimeoutException("Environment step timed out")