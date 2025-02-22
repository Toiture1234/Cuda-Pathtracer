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
		public:
			button_onoff() {
				rectangle = sf::RectangleShape({ 0.f,0.f });
				rectangle.setSize({ 0.f,0.f });
				color = sf::Color::Black;
				rectangle.setFillColor(color);
				status = nullptr;
			}
			button_onoff(sf::Vector2f pos, sf::Vector2f size) {
				rectangle.setPosition(pos);
				rectangle.setSize(size);
				color = sf::Color::Black;
				rectangle.setFillColor(color);
				status = nullptr;
			}

			inline void setBool(bool* ref) {
				status = ref;
			}
			inline bool getStatus() const {
				return *status;
			}

			inline void update(sf::RenderWindow* window) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));

				window->draw(rectangle);
				if (mousePos.x > rectangle.getPosition().x && mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && mousePos.y > rectangle.getPosition().y && mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					*status = !(*status);
				}
			}
		};
	}
}