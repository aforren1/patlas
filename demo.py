import moderngl as mgl
import glfw
import numpy as np
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
    if not glfw.init():
        raise RuntimeError('GLFW failed.')
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
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
    tex = ctx.texture(atlas.shape[0:2], atlas.shape[2], atlas)

    vbo = np.empty(4, dtype=[('vertices', np.float32, 2), ('texcoord', np.float32, 2)])
    vbo['vertices'] = [(-1, -1), (-1, -0.25), (-0.25, -1), (-0.25, -0.25)]
    vbo['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    buf = ctx.buffer(vbo)
    vao = ctx.vertex_array(prog, buf, 'vertices', 'texcoord')

    a = ap.metadata['alex']
    atex = np.array([(a['u0'], a['v0']), (a['u0'], a['v1']),
                     (a['u1'], a['v0']), (a['u1'], a['v1'])],
                     dtype='f4')
    k = ap.metadata['kazoo']
    ktex = np.array([(k['u0'], k['v0']), (k['u0'], k['v1']),
                     (k['u1'], k['v0']), (k['u1'], k['v1'])],
                     dtype='f4')
    tex.use()
    while not glfw.window_should_close(win):
        ctx.clear(0.2, 0.1, 0.1)
        # cat 1
        prog['offset'] = 1, 1
        vbo['texcoord'] = atex
        buf.write(vbo)
        vao.render(mgl.TRIANGLE_STRIP)
        # cat 2
        prog['offset'] = 0.5, 0
        vbo['texcoord'] = ktex
        buf.write(vbo)
        vao.render(mgl.TRIANGLE_STRIP)
        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()
