# Launch shopping
docker run --name shopping -p 8082:80 -d shopping_final_0712
sleep 15
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:8082" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:8082" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush

# Launch shopping_admin
docker run --name shopping_admin -p 8083:80 -d shopping_admin_final_0719
sleep 15 
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:8083" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:8083/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flu

# Launch forum
docker run --name forum -p 8080:80 -d postmill-populated-exposed-withimg

# Launch Gitlab
docker run --name gitlab --shm-size=128g -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
sleep 180
docker exec gitlab sed -i "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure

# Launch Wikipedia
docker run -d --name=wikipedia --volume=/home/zhoujun/docker_images/:/data -p 8081:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
