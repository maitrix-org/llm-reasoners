(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g i f)
(:init 
(harmony)
(planet g)
(planet i)
(planet f)
(province g)
(province i)
(province f)
)
(:goal
(and
(craves g i)
(craves i f)
)))