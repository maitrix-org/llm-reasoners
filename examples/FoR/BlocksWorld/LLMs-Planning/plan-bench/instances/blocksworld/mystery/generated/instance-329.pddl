(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l k e g)
(:init 
(harmony)
(planet l)
(planet k)
(planet e)
(planet g)
(province l)
(province k)
(province e)
(province g)
)
(:goal
(and
(craves l k)
(craves k e)
(craves e g)
)))