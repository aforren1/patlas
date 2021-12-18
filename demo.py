import moderngl as mgl
import glfw
import numpy as np
import glm
from patlas import AtlasPacker

image_vert = """
#version 330
uniform vec2 offset;
in vec2 vertices;
in vec2 texcoord;
out vec2 v_texcoord;
void main()
{
    gl_Position = vec4(vertices + offset, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

image_frag = """
#version 330
uniform sampler2D texture;
in vec2 v_texcoord;
out vec4 f_color;
void main()
{
    f_color = texture2D(texture, v_texcoord);
}
"""

if __name__ == '__main__':
    import sys

    if not glfw.init():
        raise RuntimeError('GLFW failed.')
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, True)

    if not (win := glfw.create_window(600, 600, 'Test', None, None)):
        glfw.terminate()
        raise RuntimeError('Window failed.')
    
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    ctx = mgl.create_context(require=330)

    prog = ctx.program(vertex_shader=image_vert, fragment_shader=image_frag)

    ap = AtlasPacker(2048, pad=1)

    ap.pack(['images/alex.png', 'images/kazoo.jpg'])

    atlas = ap.atlas
    ax, ay = atlas.shape[0:2]
    tex = ctx.texture(atlas.shape[0:2], atlas.shape[2], atlas)

    tmp = np.empty(4, dtype=[('vertices', np.float32, 2), ('texcoord', np.float32, 2)])
    tmp['vertices'] = [(-1, -1), (-1, -0.25), (-0.25, -1), (-0.25, -0.25)]
    tmp['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    buf = ctx.buffer(tmp)
    vao = ctx.vertex_array(prog, buf, 'vertices', 'texcoord')

    a = ap.locations['alex']
    atex = np.array([(a['x'] / ax, a['y'] / ay),
                     (a['x'] / ax, (a['y'] + a['h']) / ay),
                     ((a['x'] + a['w']) / ax, a['y'] / ay),
                     ((a['x'] + a['w']) / ax, (a['y'] + a['h']) / ay)], 
                     dtype='f4')
    k = ap.locations['kazoo']
    ktex = np.array([(k['x'] / ax, k['y'] / ay),
                     (k['x'] / ax, (k['y'] + k['h']) / ay),
                     ((k['x'] + k['w']) / ax, k['y'] / ay),
                     ((k['x'] + k['w']) / ax, (k['y'] + k['h']) / ay)], 
                     dtype='f4')
    tex.use()
    while not glfw.window_should_close(win):
        ctx.clear(0.2, 0.1, 0.1)
        # cat 1
        prog['offset'].write(glm.vec2(1, 1))
        tmp['texcoord'] = atex
        buf.write(tmp)
        vao.render(mgl.TRIANGLE_STRIP)

        # cat 2
        prog['offset'].write(glm.vec2(0.5, 0))
        tmp['texcoord'] = ktex
        buf.write(tmp)
        vao.render(mgl.TRIANGLE_STRIP)
        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()
