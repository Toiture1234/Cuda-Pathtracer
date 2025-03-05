#pragma once

namespace pathtracer {
	namespace Buttons {
		class button_onePress {
		private :
			sf::RectangleShape rectangle;
			sf::Color color;

			sf::Sprite rectangleSprite;
			sf::Texture rectTexture;

			float lastpressedTime;
		public:
			button_onePress() {
				rectangle = sf::RectangleShape({ 0.f,0.f });
				rectangle.setSize({ 0.f,0.f });
				color = sf::Color::Black;
				lastpressedTime = 0.f;
			}
			button_onePress(sf::Vector2f pos, sf::Vector2f size) {
				rectangle.setPosition(pos);
				rectangle.setSize(size);
				lastpressedTime = 0.f;
			}

			inline void setTex(sf::Texture* tex) {
				rectTexture = *tex;
				rectangleSprite.setTexture(rectTexture);
				rectangleSprite.setScale({ rectangle.getSize().x / tex->getSize().x, rectangle.getSize().y / tex->getSize().y });
				rectangleSprite.setPosition(rectangle.getPosition());
			}

			inline bool isPressed(sf::RenderWindow* window, float time) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));
				

				if (mousePos.x > rectangle.getPosition().x && 
					mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && 
					mousePos.y > rectangle.getPosition().y && 
					mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && 
					sf::Mouse::isButtonPressed(sf::Mouse::Left) &&
					abs(lastpressedTime - time) > 0.4f)
				{
					lastpressedTime = time;
					return true;
				}
				return false;
			}
			inline void draw(sf::RenderWindow* window) {
				window->draw(rectangleSprite);
			}
		};
		class button_onoff { // need to add a cooldown
		private:
			sf::RectangleShape rectangle;
			sf::Color color;

			sf::Sprite rectangleSprite;
			sf::Texture rectTexture;

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

			inline void setTex(sf::Texture* tex) {
				rectTexture = *tex;
				rectangleSprite.setTexture(rectTexture);
				rectangleSprite.setScale({ rectangle.getSize().x / tex->getSize().x, rectangle.getSize().y / tex->getSize().y });
				rectangleSprite.setPosition(rectangle.getPosition());
			}

			inline void setBool(bool* ref) {
				status = ref;
			}
			inline bool getStatus() const {
				return *status;
			}

			inline bool update(sf::RenderWindow* window, float time) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));

				//window->draw(rectangle);
				window->draw(rectangleSprite);
				if (mousePos.x > rectangle.getPosition().x && 
					mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && 
					mousePos.y > rectangle.getPosition().y && 
					mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && 
					sf::Mouse::isButtonPressed(sf::Mouse::Left) && 
					abs(lastpressedTime - time) > 0.4f) 
				{
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

			sf::Sprite rectangleSprite;
			sf::Texture rectTexture;

			float minValue, maxValue;
			float* ref;
			float delta;
		public:
			button_slider() {
				rectangle = sf::RectangleShape({ 0.f,0.f });
				rectangle.setSize({ 0.f,0.f });
				rectangle.setFillColor({ 88,109,120 });
				inside_rect.setFillColor({ 58,71,79 });
				slider.setFillColor({ 142,159,169 });
				slider = sf::CircleShape(0.f);
				ref = nullptr;
				minValue = maxValue = 0.f;
				delta = 0.f;
			}
			button_slider(sf::Vector2f pos, sf::Vector2f size) {
				rectangle.setPosition(pos);
				rectangle.setSize(size);
				inside_rect.setPosition({ pos.x + size.x * 0.1f, pos.y + size.y * 0.4f });
				inside_rect.setSize({ size.x * 0.8f, size.y * 0.2f });
				slider.setRadius(size.y * 0.3f);
				slider.setPosition({ pos.x + size.x * 0.1f, pos.y + size.y * 0.2f}); // inited to 0
				
				rectangle.setFillColor({ 88,109,120 });
				inside_rect.setFillColor({ 58,71,79 });
				slider.setFillColor({ 142,159,169 });
				ref = nullptr;
				minValue = maxValue = 0.f;
				delta = 0.f;
			}

			inline void setTex(sf::Texture* tex) {
				rectTexture = *tex;
				rectangleSprite.setTexture(rectTexture);

				sf::Vector2f size = rectangle.getSize();
				sf::Vector2f pos = rectangle.getPosition();
				rectangleSprite.setScale({ size.x / tex->getSize().x, size.y / tex->getSize().y });
				rectangleSprite.setPosition(pos);
			}

			inline void setRef(float* ref0, float mi, float ma) {
				ref = ref0, minValue = mi, maxValue = ma;
				*ref = clamp(*ref, mi, ma);
				delta = ma - mi;

				// max position is given by pos.x + size.x - size.y * 0.6f - size.x * 0.1f
				// = pos.x + size.x(1.f - 0.1f) - size.y * 0.6f
				// = pos.x + size.x * 0.9 - size.y * 0.6f
				// so delta in coord is : pos.x + size.x * 0.9f - size.y * 0.6f - pos.x - size.x * 0.1f
				// = size.x * 0.8f - size.y * 0.6f
				// final position is given by : pos.x + size.x * 0.1f + (size.x * 0.8f - size.y * 0.6f) * p , where p is a parameter from 0 to 1
				sf::Vector2f size = rectangle.getSize();
				sf::Vector2f pos = rectangle.getPosition();
				slider.setPosition({ pos.x + size.x * 0.1f + (size.x * 0.8f - size.y * 0.6f) * ((*ref - minValue) / delta), slider.getPosition().y});
			}

			inline bool update(sf::RenderWindow* window) {
				sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));

				//window->draw(rectangle);
				window->draw(rectangleSprite);
				window->draw(inside_rect);
				window->draw(slider);
				if (mousePos.x > rectangle.getPosition().x && 
					mousePos.x < rectangle.getPosition().x + rectangle.getSize().x && 
					mousePos.y > rectangle.getPosition().y && 
					mousePos.y < rectangle.getPosition().y + rectangle.getSize().y && 
					sf::Mouse::isButtonPressed(sf::Mouse::Left)) 
				{
					sf::Vector2f size = rectangle.getSize();
					sf::Vector2f pos = rectangle.getPosition();
					float xCoord = (float)(mousePos.x - pos.x - size.x * 0.1f - size.y * 0.3f) / (float)(size.x * 0.8f - size.y * 0.6f);
					xCoord = clamp(xCoord, 0.f, 1.f);
					*ref = minValue + xCoord * delta;
					slider.setPosition({ pos.x + size.x * 0.1f + (size.x * 0.8f - size.y * 0.6f) * ((*ref - minValue) / delta), slider.getPosition().y });
					return true;
				}
				return false;
			}
		};
	}
}