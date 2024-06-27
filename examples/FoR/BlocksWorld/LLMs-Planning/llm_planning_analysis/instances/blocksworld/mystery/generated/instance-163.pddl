(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k e i h)
(:init 
(harmony)
(planet k)
(planet e)
(planet i)
(planet h)
(province k)
(province e)
(province i)
(province h)
)
(:goal
(and
(craves k e)
(craves e i)
(craves i h)
)))