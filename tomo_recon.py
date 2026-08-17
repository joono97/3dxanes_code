import numpy as np
import skimage.restoration as skr
import numexpr as ne
import napari, time, h5py, os, tomopy, tifffile, re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button
import glob
from scipy.ndimage import zoom, gaussian_filter as gf, median_filter as median


matplotlib.use('QT5Agg')
ncore = os.cpu_count() // 2


class VolumeViewer(object):
    def __init__(self, img_stack, nframes, center, fn):
        self.img_stack = img_stack
        self.nframes = nframes
        self.center = center
        self.fn = fn
        self.brightness = 1.0
        self.contrast = 0.0

        # Setup the axes.
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)  # Make room for sliders
        self.slider_ax = self.fig.add_axes((0.2, 0.03, 0.55, 0.03))
        self.register_ax = self.fig.add_axes((0.85, 0.72, 0.1, 0.04))

        # Brightness and contrast sliders
        self.brightness_ax = self.fig.add_axes((0.25, 0.10, 0.65, 0.02))
        self.contrast_ax = self.fig.add_axes((0.25, 0.06, 0.65, 0.02))

        self.brightness_slider = Slider(self.brightness_ax, 'Brightness', 0.1, 3.0, valinit=1.0)
        self.contrast_slider = Slider(self.contrast_ax, 'Contrast', -1.0, 1.0, valinit=0.0)

        # Connect brightness/contrast sliders
        self.brightness_slider.on_changed(self.update_brightness_contrast)
        self.contrast_slider.on_changed(self.update_brightness_contrast)

        # Make the slider
        self.slider = Slider(self.slider_ax, 'Frame', 1, self.nframes,
                             valinit=1, valfmt='%1d/{}'.format(self.nframes))
        self.slider.on_changed(self.update)

        # Make the buttons
        self.reg_button = Button(self.register_ax, 'Register')
        self.reg_button.on_clicked(self.img_update)

        # Plot the first slice of the image
        self.frame = 0
        img = self.apply_brightness_contrast(np.array(self.img_stack[self.frame]))
        self.im = self.ax.imshow(img, cmap='gray')

    def apply_brightness_contrast(self, image):
        """Apply brightness and contrast to the image."""
        return np.clip(self.brightness * image + self.contrast * 255, 0, 255)

    def update_brightness_contrast(self, value):
        self.brightness = self.brightness_slider.val
        self.contrast = self.contrast_slider.val
        self.update(self.slider.val)

    def update(self, value):
        self.frame = int(np.round(value - 1))
        dat = np.array(self.img_stack[self.frame])
        dat = self.apply_brightness_contrast(dat)
        self.im.set_data(dat)
        self.im.set_clim((dat.min(), dat.max()))
        self.fig.canvas.draw()

    def img_update(self, event):
        print(f"{self.fn.split('/')[-1]} Rotation center: {self.center[self.frame]}")
        dir = os.path.dirname(self.fn)
        scan_no = re.findall(r'\d+', self.fn.split('/')[-1])[0]
        f = open(dir + f"/rotcen_{self.fn.split('/')[-3]}.txt", 'a')
        f.write(f"{scan_no}\t{self.center[self.frame]}\n")
        f.close()

    def show(self):
        plt.show()

def denoise(img, denoise_type):
    if denoise_type == 'wiener':
        psf = np.ones([2, 2]) / (2 ** 2)
        for k in range(img.shape[0]):
            img[k] = skr.wiener(img[k], psf=psf, reg=None, balance=0.3, is_real=True, clip=True)[:]
    elif denoise_type == "nl_means":
        for k in range(img.shape[0]):
            img[k] = skr.denoise_nl_means(img[k], patch_size=5, patch_distance=7, h=0.1, multichannel=False,
                                          fast_mode=True, sigma=0.05, preserve_range=False, channel_axis=None)
    elif denoise_type == 'median':
        img[:] = median(img, size=(1, 5, 5))[:]

    elif denoise_type == "tv_bregman":
        for k in range(img.shape[0]):
            img[k] = skr.denoise_tv_bregman(img[k], weight=1.0, max_num_iter=100, eps=0.01,
                                            isotropic=True, channel_axis=False)
    elif denoise_type == "tv_chambolle":
        for k in range(img.shape[0]):
            img[k] = skr.denoise_tv_chambolle(img[k], weight=0.1, max_num_iter=100, eps=0.002, channel_axis=False)
    elif denoise_type == "bilateral":
        for k in range(img.shape[0]):
            img[k] = skr.denoise_bilateral(img[k], win_size=None, sigma_color=None, sigma_spatial=1, bins=10000,
                                           mode='constant', cval=0, channel_axis=None)
            "mode: How to handle values outside the img borders. See numpy.pad for detail"
    elif denoise_type == "wavelet":
        for k in range(img.shape[0]):
            img[k] = skr.denoise_wavelet(img[k], sigma=1, wavelet='db1', mode='soft', wavelet_levels=3,
                                         convert2ycbcr=False, method='BayesShrink',
                                         rescale_sigma=True, channel_axis=None)
    elif denoise_type == 'gaussian':
        from skimage.filters import gaussian as gf
        img = gf(img, [0, 1, 1])
    return img


def rmv_stripe(img, stripe_type):
    if stripe_type == 'all_stripe':
        img = tomopy.prep.stripe.remove_all_stripe(img, snr=3, la_size=81, sm_size=21, dim=2, ncore=ncore)
    elif stripe_type == 'fw_stripe':
        img = tomopy.prep.stripe.remove_stripe_fw(img, level=8, wname='db5', sigma=2, pad=True, ncore=ncore)
    elif stripe_type == 'ti_stripe':
        img = tomopy.prep.stripe.remove_stripe_ti(img, nblock=0, alpha=1.5, ncore=ncore)
    elif stripe_type == 'sf_stripe':
        img = tomopy.prep.stripe.remove_stripe_sf(img, size=31, ncore=ncore)
    elif stripe_type == "vo_interp_stripe":
        from algotom.prep import removal
        img = tomopy.prep.stripe.remove_stripe_based_interpolation(img, snr=3.0, size=51, drop_ratio=0.1, norm=True)
    return img


def normalize(arr, flat, dark, cutoff=None, ncore=os.cpu_count() // 2):
    flat = np.mean(flat, axis=0, dtype=np.float32)
    dark = np.mean(dark, axis=0, dtype=np.float32)
    with tomopy.util.mproc.set_numexpr_threads(ncore):
        denom = (flat - dark).astype(np.float32)
        out = (arr - dark).astype(np.float32)
        out[:] = (out / denom)[:]
        out[np.isnan(out)] = 1
        out[np.isinf(out)] = 1
        out[out <= 0] = 1
        if cutoff is not None:
            cutoff = np.float32(cutoff)
            out[:] = np.where(out > cutoff, cutoff, out)[:]
    return tomopy.prep.normalize.minus_log(out, ncore=os.cpu_count() // 2)


def load_file(fn, sli):
    if len(sli) == 0:
        with h5py.File(fn, 'r') as f:
            proj = np.array(f['img_tomo'])
            bkg = np.array(f['img_bkg_avg'])
            dark = np.array(f['img_dark_avg'])
            theta = np.array(f["angle"]) / 180.0 * np.pi
            scan_id = np.array(f["scan_id"])
    else:
        with h5py.File(fn, 'r') as f:
            proj = np.array(f['img_tomo'][:, sli[0]: sli[1], :])
            bkg = np.array(f['img_bkg_avg'][:, sli[0]: sli[1]])
            dark = np.array(f['img_dark_avg'][:, sli[0]: sli[1]])
            theta = np.array(f["angle"]) / 180.0 * np.pi
            scan_id = np.array(f["scan_id"])
    return proj, bkg, dark, theta, scan_id


def rotcen(fn, sli=None, cen_range=(450, 550, 1)): #회전중심 설정해야함 샘플마다 조금씩 구간다르게 해야함.
    print(f"\nLoading images: {fn.split('/')[-1]}")
    img, bkg, dark, theta, scan_id = load_file(fn, sli=[300, 700])
    print("Normalizing image..")
    norm = normalize(img, bkg, dark)
    print(f"Removing stripe...")
    norm = rmv_stripe(norm, stripe_type="all_stripe")
    print(f"Denoising...")
    norm = denoise(norm, denoise_type="gaussian")

    if cen_range is None:
        center = np.arange(norm.shape[2] / 2 - 2, norm.shape[2] / 2 + 2, 0.5)
    else:
        center = np.arange(*cen_range)
    if sli is None:
        ind = norm.shape[1] // 4
    else:
        ind = sli
    stack = tomopy.util.dtype.empty_shared_array((len(center), norm.shape[0], norm.shape[2]))
    for m in range(center.size):
        stack[m] = norm[:, ind, :]

    rec = tomopy.recon(stack, theta, center=center, sinogram_order=True, algorithm='gridrec',
                       filter_name='hamming', nchunk=1, ncore=ncore)
    viewer = VolumeViewer(rec, len(rec), center, fn)
    viewer.show()
    return rec


def recon(fn, rot_cen, nchunk, sli, col=[], stripe_type='all_stripe', denoise_type="gaussian"):
    time_s = time.time()
    print(f"Loading images: {fn.split('/')[-1]}")
    img, bkg, dark, theta, scan_id = load_file(fn, sli)
    print("Normalizing image..")
    norm = normalize(img, bkg, dark)
    print(f"Removing stripe: {stripe_type}")
    norm = rmv_stripe(norm, stripe_type= stripe_type)
    print(f"Denoising: {denoise_type}")
    norm = denoise(norm, denoise_type=denoise_type)
    print(f"Reconstructing {fn.split('/')[-1]}")
    data_recon = tomopy.recon(norm, theta, center=rot_cen, algorithm='gridrec', ncore=ncore, filter_name='hamming'),
    '''
    ratio = 0.9
    print(f"Circular mask : {ratio}")
    data_recon = tomopy.circ_mask(data_recon[0], 0, ratio=ratio)
    '''
    time_e = time.time()
    cwd = os.path.dirname(fn)
    try:
        os.mkdir(cwd + '/recon_' + fn.split('/')[-1][:-3])
    except:
        print(cwd + '/recon_' + fn.split('/')[-1][:-3] + ' existed')
    fnt = cwd + '/recon_' + fn.split('/')[-1][:-3] + '/recon_' + fn.split('/')[-1][:-3] + '_{}.tiff'
    print("Saving Tiff files")
    write_tiff_stack(data_recon[0] * 100, axis=0, fnt=fnt, start=sli[0], overwrite=True)
    print(f'Recon takes {time_e - time_s:3.1f} seconds')


def write_tiff_stack(img_stack, axis=0, fnt=None, start=0, overwrite=True):
    if axis == 0:
        for ii in range(img_stack.shape[axis]):
            tifffile.imwrite(fnt.format(str(start + ii).zfill(5)), img_stack[ii].astype(np.float32))
    elif axis == 1:
        for ii in range(img_stack.shape[axis]):
            tifffile.imwrite(fnt.format(str(start + ii).zfill(5)), img_stack[:, ii, :].astype(np.float32))
    elif axis == 2:
        for ii in range(img_stack.shape[axis]):
            tifffile.imwrite(fnt.format(str(start + ii).zfill(5)), img_stack[:, :, ii].astype(np.float32))


if __name__ == '__main__':
    tomo_dir = '/media/joonho/2026 BNL/2026BNL/SC6 DOD13 0.1C Q_2'
    flist = sorted(glob.glob(tomo_dir + "/fly_scan_id_*.h5"))
    for i in range(0,len(flist)):
        #rec =   rotcen(flist[i], sli=250)
        rot_cen = np.loadtxt(tomo_dir + '/rotcen_2026BNL.txt')
        data_recon = recon(flist[i], rot_cen=rot_cen[:,1][i], nchunk=200, sli=[1, 600])
