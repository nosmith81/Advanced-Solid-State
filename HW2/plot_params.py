import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def calc_etas(R, d0):
    """Calculate overlap parameters"""
    eta11 = (1 + R/d0 + R**2/(3*d0**2)) * np.exp(-R/d0)
    eta12 = 0.5 * (1 + R/d0 + R**2/(64*d0**2)) * np.exp(-R/(2*d0)) - 0.5 * np.exp(-R/d0)
    eta22 = (1 + R/(2*d0) + R**2/(24*d0**2)) * np.exp(-R/(2*d0))
    return eta11, eta12, eta22

def calc_ts(R, d0, t0=1.0):
    """Calculate hopping parameters"""
    # Note: For R < 3*d0, these functions might not accurately represent the physical system 
    # as per the problem description, but they can be evaluated mathematically.
    # Also, t_nm(0) = 0 is a special condition given in the text, which isn't explicitly 
    # matched by evaluating the functions below at R=0, so this formula is primarily for R > 3*d0.
    t11 = -t0 * (1 + 2*R/d0) * np.exp(-R/d0)
    t12 = -t0 * (1 + 2*R/d0) * np.exp(-R/d0) + t0 * (1 + R/(16*d0)) * np.exp(-R/(2*d0))
    t22 = -t0 * (1 + R/(2*d0) - R**2/(24*d0**2)) * np.exp(-R/(2*d0))
    
    # Enforce t_nm(0) = 0 as given in the problem statement
    # We apply a small mask for R values very close to 0
    mask = R < 1e-10
    t11[mask] = 0
    t12[mask] = 0
    t22[mask] = 0
    
    return t11, t12, t22

def main():
    # Initial parameters
    d0_init = 1.0
    t0_val = 1.0 # use t0 as the energy unit
    
    # Define a range for R
    # The expressions are valid for R > 3*d0
    # Let's plot from 0 to 25 to see the behavior
    R = np.linspace(0.0, 25.0, 1000)
    
    # Setup the figure and subplots
    fig, (ax_eta, ax_t) = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    
    # Initial data
    eta11, eta12, eta22 = calc_etas(R, d0_init)
    t11, t12, t22 = calc_ts(R, d0_init, t0_val)
    
    # Plot eta
    l_eta11, = ax_eta.plot(R, eta11, label=r'$\eta_{11}$')
    l_eta12, = ax_eta.plot(R, eta12, label=r'$\eta_{12}$')
    l_eta22, = ax_eta.plot(R, eta22, label=r'$\eta_{22}$')
    ax_eta.set_ylabel(r'Overlap Integrals $\eta(R)$')
    ax_eta.set_title('Dependence of Overlap Parameters on Distance')
    ax_eta.legend()
    ax_eta.grid(True)
    
    # Plot t
    l_t11, = ax_t.plot(R, t11, label=r'$t_{11}$')
    l_t12, = ax_t.plot(R, t12, label=r'$t_{12}$')
    l_t22, = ax_t.plot(R, t22, label=r'$t_{22}$')
    ax_t.set_ylabel(r'Hopping Parameters $t(R)$ (units of $t_0$)')
    ax_t.set_xlabel('Distance $R$')
    ax_t.set_title('Dependence of Hopping Parameters on Distance')
    ax_t.legend()
    ax_t.grid(True)
    
    # Create the slider axis and slider
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    d0_slider = Slider(
        ax=ax_slider,
        label=r'$d_0$',
        valmin=0.1,
        valmax=5.0,
        valinit=d0_init,
        valstep=0.1
    )
    
    # Update function for the slider
    def update(val):
        d0_val = d0_slider.val
        
        # Recalculate
        new_eta11, new_eta12, new_eta22 = calc_etas(R, d0_val)
        new_t11, new_t12, new_t22 = calc_ts(R, d0_val, t0_val)
        
        # Update data
        l_eta11.set_ydata(new_eta11)
        l_eta12.set_ydata(new_eta12)
        l_eta22.set_ydata(new_eta22)
        
        l_t11.set_ydata(new_t11)
        l_t12.set_ydata(new_t12)
        l_t22.set_ydata(new_t22)
        
        # Dynamically rescale y-axis to fit new data
        ax_eta.relim()
        ax_eta.autoscale_view()
        ax_t.relim()
        ax_t.autoscale_view()
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    d0_slider.on_changed(update)
    
    # Add a vertical line/shade to indicate R > 3*d0 validity region for initial d0
    valid_line_eta = ax_eta.axvline(3*d0_init, color='red', linestyle='--', alpha=0.5, label=r'Validity limit ($R=3d_0$)')
    valid_line_t = ax_t.axvline(3*d0_init, color='red', linestyle='--', alpha=0.5, label=r'Validity limit ($R=3d_0$)')
    
    # We need to handle the update of the vertical line as well
    def update_validity_lines(val):
        valid_line_eta.set_xdata([3*val, 3*val])
        valid_line_t.set_xdata([3*val, 3*val])
        update(val)
        
    d0_slider.on_changed(update_validity_lines)
    
    plt.show()

if __name__ == '__main__':
    main()
