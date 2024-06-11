(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g b k)
(:init 
(harmony)
(planet g)
(planet b)
(planet k)
(province g)
(province b)
(province k)
)
(:goal
(and
(craves g b)
(craves b k)
)))