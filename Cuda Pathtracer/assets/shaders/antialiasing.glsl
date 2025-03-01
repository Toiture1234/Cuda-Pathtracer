uniform sampler2D texture;
uniform vec2 resolution;

// color transformations
uniform vec3 multiplier;
uniform float gamma;
uniform float contrast;
uniform float saturation;

void main()
{
    // lookup the pixel in the texture
    vec2 uv = gl_TexCoord[0].xy;
    ivec2 iuv = ivec2(uv * resolution);
    vec4 pixel = texture2D(texture, uv);

    // antialiasing
    vec3 totalCol = pixel.rgb;

    totalCol += texelFetch(texture, iuv + ivec2(0, 1), 0).rgb * 0.33f;
    totalCol += texelFetch(texture, iuv + ivec2(0, -1), 0).rgb * 0.33f;
    totalCol += texelFetch(texture, iuv + ivec2(1, 0), 0).rgb * 0.33f;
    totalCol += texelFetch(texture, iuv + ivec2(-1, 0), 0).rgb * 0.33f;

    totalCol /= 0.33f * 4.f + 1.f;
    // multiply it by the color
    gl_FragColor = gl_Color * vec4(totalCol, 1.);
}