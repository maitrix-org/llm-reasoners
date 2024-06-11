(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e g c h)
(:init 
(harmony)
(planet e)
(planet g)
(planet c)
(planet h)
(province e)
(province g)
(province c)
(province h)
)
(:goal
(and
(craves e g)
(craves g c)
(craves c h)
)))