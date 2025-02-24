#pragma once

namespace pathtracer {
	namespace Buttons {
		class button_onePress {
		private :
			sf::RectangleShape rectangle;
			sf::Color color;

		public:
			button_onePress() {
				rectangle = sf::RectangleShape({ 0.f,0.f });
				rectangle.setSize({ 0.f,0.f });
				color = sf::Color::Black;
			}
			button_onePress(sf::Vector2f pos, sf::Vector2f size) {
				rectangle.setPosition(pos);
				rectangle.setSize(size);
			}

			inline bool isPressed(sf::RenderWindow* window) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));

				if (mousePos.x > rectangle.getPosition().x && mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && mousePos.y > rectangle.getPosition().y && mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					return true;
				}
				return false;
			}
		};
		class button_onoff { // need to add a cooldown
		private:
			sf::RectangleShape rectangle;
			sf::Color color;

			bool* status;
			float lastpressedTime;
		public:
			button_onoff() {
				rectangle = sf::RectangleShape({ 0.f,0.f });
				rectangle.setSize({ 0.f,0.f });
				color = sf::Color::Black;
				rectangle.setFillColor(color);
				status = nullptr;
				lastpressedTime = 0.f;
			}
			button_onoff(sf::Vector2f pos, sf::Vector2f size) {
				rectangle.setPosition(pos);
				rectangle.setSize(size);
				color = sf::Color::Black;
				rectangle.setFillColor(color);
				status = nullptr;
				lastpressedTime = 0.f;
			}

			inline void setBool(bool* ref) {
				status = ref;
			}
			inline bool getStatus() const {
				return *status;
			}

			inline bool update(sf::RenderWindow* window, float time) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));

				window->draw(rectangle);
				if (mousePos.x > rectangle.getPosition().x && mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && mousePos.y > rectangle.getPosition().y && mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && sf::Mouse::isButtonPressed(sf::Mouse::Left) && abs(lastpressedTime - time) > 0.4f) {
					*status = !(*status);
					lastpressedTime = time;
					return true;
				}
				return false;
			}
		};
		class button_slider {
		private:
			sf::RectangleShape rectangle;
			sf::RectangleShape inside_rect;
			sf::CircleShape slider;
			sf::Color color0;
			sf::Color color1;
			sf::Color color2;

			float minValue, maxValue;
			float* ref;
			float delta;
		public:
			button_slider() {
				rectangle = sf::RectangleShape({ 0.f,0.f });
				rectangle.setSize({ 0.f,0.f });
				color0 = color1 = color2 = sf::Color::Black;
				rectangle.setFillColor(color0);
				slider = sf::CircleShape(0.f);
				ref = nullptr;
				minValue = maxValue = 0.f;
				delta = 0.f;
			}
			button_slider(sf::Vector2f pos, sf::Vector2f size) {
				rectangle.setPosition(pos);
				rectangle.setSize(size);
				inside_rect.setPosition(sf::Vector2f(pos.x + size.x * 0.1f, pos.y + size.y * 0.25f));
				inside_rect.setSize(sf::Vector2f(size.x * 0.8f, size.y * 0.5f));
				slider.setPosition(sf::Vector2f(pos.x + size.x * 0.1f, pos.y + size.y * 0.25f));
				slider.setRadius(size.y * 0.5f);

				rectangle.setFillColor(sf::Color::Black);
				inside_rect.setFillColor(sf::Color::Green);
				slider.setFillColor(sf::Color::White);
				ref = nullptr;
				minValue = maxValue = 0.f;
				delta = 0.f;
			}

			inline void setRef(float* ref0, float mi, float ma) {
				ref = ref0, minValue = mi, maxValue = ma;
				*ref = clamp(*ref, mi, ma);
				delta = ma - mi;
			}

			inline void update(sf::RenderWindow* window) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));

				window->draw(rectangle);
				window->draw(inside_rect);
				window->draw(slider);
				if (mousePos.x > rectangle.getPosition().x && mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && mousePos.y > rectangle.getPosition().y && mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					float xCoord = (float)(mousePos.x - inside_rect.getPosition().x) / (float)inside_rect.getSize().x;
					xCoord = clamp(xCoord, 0.f, 1.f);
					*ref = minValue + xCoord * delta;
					slider.setPosition({ mousePos.x, slider.getPosition().y });
				}
			}
		};
	}
}