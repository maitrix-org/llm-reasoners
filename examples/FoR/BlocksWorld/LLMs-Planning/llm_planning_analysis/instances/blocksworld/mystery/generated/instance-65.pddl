(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d k e)
(:init 
(harmony)
(planet d)
(planet k)
(planet e)
(province d)
(province k)
(province e)
)
(:goal
(and
(craves d k)
(craves k e)
)))