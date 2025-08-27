import carla
import time

def main():
    
    try:
        SpawnActor = carla.command.SpawnActor
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.load_world('Town01')

        distance = 10 #waypoint的间距
        waypoints = world.get_map().generate_waypoints(distance)
        for w in waypoints:
            world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                            persistent_lines=True)
        while True:
            world.wait_for_tick()
        
    finally:
        print('\ndestroying vehicles')
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
