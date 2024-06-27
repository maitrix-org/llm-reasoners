(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i k l f)
(:init 
(harmony)
(planet i)
(planet k)
(planet l)
(planet f)
(province i)
(province k)
(province l)
(province f)
)
(:goal
(and
(craves i k)
(craves k l)
(craves l f)
)))