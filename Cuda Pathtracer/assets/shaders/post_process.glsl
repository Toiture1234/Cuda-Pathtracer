uniform sampler2D texture;
uniform vec2 resolution;

// color transformations
uniform vec3 multiplier;
uniform float gamma;
uniform float contrast;
uniform float saturation;
uniform float exposure;

void main()
{
    // lookup the pixel in the texture
    vec2 uv = gl_TexCoord[0].xy;
    ivec2 iuv = ivec2(uv * resolution);
    vec4 pixel = texture2D(texture, uv);
    vec3 col = pixel.rgb;
    
    col *= exposure;
    col *= multiplier;
    float lw = dot(vec3(0.2126, 0.7152, 0.0722), col);
    col = mix(vec3(lw), col, saturation);
    col = pow(col, vec3(gamma));

    // multiply it by the color
    gl_FragColor = gl_Color * vec4(col, 1.0);
}