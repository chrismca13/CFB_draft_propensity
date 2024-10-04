import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors 


theme_color_names = {
                        'maize': '#FFCB05',
                        'light_maize': '#FFE066',
                        'white': '#FFFFFF',
                        'light_blue': '#7DAFDB', 
                        'blue': '#00274C'
                        }

def create_custom_palette():
    colors_dict = theme_color_names.copy()
    colors_dict.pop('white', None)
    custom_palette= list(colors_dict.values())
    return custom_palette

def create_custom_cmap(hex_colors):
    '''
    Create a custom colormap from a list of hex colors.
    Parameters: 
        hex_colors: string hex value 
    '''
    rgb_colors = [mcolors.hex2color(c) for c in hex_colors]    # Convert hex to RGB using matplotlib.colors
    return LinearSegmentedColormap.from_list("custom_cmap", rgb_colors)     # Create and return the custom colormap

def generate_cmap_theme(verbose=False): 
    '''
    Generate custom colormap from a list of hex colors.
    Parameters: 
        verbose: boolean to determine whether we would like to view the colors
    '''
    custom_cmap = create_custom_cmap(list(theme_color_names.values()))
    if verbose: 
        plt.figure(figsize=(6, 1))
        plt.imshow([list(range(10))], cmap=custom_cmap, aspect='auto')
        plt.colorbar()
        plt.title('Custom Colormap Example')
        plt.show()
    return custom_cmap

def generate_hex(color_name='blue'):
    '''
    Generate hex colormap given one color name 
    Parameters: 
        verbose: boolean to determine whether we would like to view the colors
    '''
    if color_name in theme_color_names: 
        c = theme_color_names[color_name]
        rgb_color= mcolors.hex2color(c)
        return LinearSegmentedColormap.from_list("single_color_cmap", [(1, 1, 1), rgb_color])

    else: 
        print(f'Select a valid color name. {color_name} is not found within the valid column names: {list(theme_color_names.keys())}.')
        return None


