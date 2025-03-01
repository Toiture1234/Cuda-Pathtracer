#pragma once

namespace pathtracer {
	class GUI {
	private :
		// buttons
		Buttons::button_slider camSpeed;
		Buttons::button_slider dof_strength;
		Buttons::button_slider focalDist;
		Buttons::button_slider fov;

		Buttons::button_onoff toggleRender;
		

		//textures
		sf::Texture button_back;
	public:
		GUI() {}

		inline void init(kernelParams& params) {
			// sizes
			camSpeed = Buttons::button_slider({ 0.f,0.f }, { 200.f, 50.f });
			dof_strength = Buttons::button_slider({ 0.f,50.f }, { 200.f,50.f });
			focalDist = Buttons::button_slider({ 0.f,100.f }, { 200.f,50.f });
			fov = Buttons::button_slider({ 0.f,150.f }, { 200.f, 50.f });

			toggleRender = Buttons::button_onoff({ 0.f, 200.f }, { 50.f,50.f });
			changeModel = Buttons::button_onePress({ 0.f,250.f }, { 50.f,50.f });
			
			// variables
			camSpeed.setRef(&params.cameraSpeed, 120.f, 1440.f);
			dof_strength.setRef(&params.DOF_strenght, 0.f, 20.f);
			focalDist.setRef(&params.focalDistance, 10.f, 1000.f);
			fov.setRef(&params.fov, 0.5f, 5.f);

			toggleRender.setBool(&params.isRendering);

			button_back.loadFromFile("assets/gui/textures/button_back.png");
			camSpeed.setTex(&button_back);
			dof_strength.setTex(&button_back);
			focalDist.setTex(&button_back);
			fov.setTex(&button_back);

			toggleRender.setTex(&button_back);
			changeModel.setTex(&button_back);
		}

		inline void update(sf::RenderWindow* window, float time, kernelParams& params) {
			camSpeed.update(window);
			dof_strength.update(window);
			focalDist.update(window);
			fov.update(window);

			if (toggleRender.update(window, time)) params.frameIndex = 0;
			changeModel.draw(window);
		}

		Buttons::button_onePress changeModel;
	};
}