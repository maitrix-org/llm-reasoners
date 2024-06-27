


(define (problem logistics-c3-s2-p1-a2)
(:domain logistics-strips)
(:objects a0 a1 
          c0 c1 c2 
          t0 t1 t2 
          l0-0 l0-1 l1-0 l1-1 l2-0 l2-1 
          p0 
)
(:init
    (AIRPLANE a0)
    (AIRPLANE a1)
    (CITY c0)
    (CITY c1)
    (CITY c2)
    (TRUCK t0)
    (TRUCK t1)
    (TRUCK t2)
    (LOCATION l0-0)
    (in-city  l0-0 c0)
    (LOCATION l0-1)
    (in-city  l0-1 c0)
    (LOCATION l1-0)
    (in-city  l1-0 c1)
    (LOCATION l1-1)
    (in-city  l1-1 c1)
    (LOCATION l2-0)
    (in-city  l2-0 c2)
    (LOCATION l2-1)
    (in-city  l2-1 c2)
    (AIRPORT l0-0)
    (AIRPORT l1-0)
    (AIRPORT l2-0)
    (OBJ p0)
    (at t0 l0-1)
    (at t1 l1-0)
    (at t2 l2-0)
    (at p0 l2-1)
    (at a0 l1-0)
    (at a1 l0-0)
)
(:goal
    (and
        (at p0 l1-0)
    )
)
)


