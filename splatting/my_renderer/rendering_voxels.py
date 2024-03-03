import torch
import cuda_renderer

def rendering( #上级代码入口
    ndc_xy_view_zs,
    features,
    image_width,
    image_height,
    radius,
):
    return _Rendering.apply(
        ndc_xy_view_zs,
        features,
        image_width,
        image_height,
        radius,
    )
    
class _Rendering(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coordinates,
        features,
        image_width,
        image_height,
        radius
    ):
        args = (
            coordinates,
            features,
            image_width,
            image_height,
            radius,
        )

        num_rendered, color, geomBuffer, binningBuffer, imgBuffer = cuda_renderer.rendering(*args)

        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.radius = radius
        ctx.save_for_backward(coordinates, features, geomBuffer, binningBuffer, imgBuffer)
        return color

    @staticmethod
    def backward(ctx, grad_out_color):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        radius = ctx.radius
        coordinates, features, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (coordinates, 
                features, 
                grad_out_color, 
                geomBuffer,
                num_rendered,
                radius,
                binningBuffer,
                imgBuffer)

        grad_coordinates, grad_features = cuda_renderer.rendering_backward(*args)

        grads = (
            grad_coordinates,
            grad_features,
            None,
            None,
            None
        )

        return grads
