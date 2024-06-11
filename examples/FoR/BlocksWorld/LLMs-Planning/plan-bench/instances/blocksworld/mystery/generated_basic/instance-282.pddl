

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b a)
(craves c b)
(planet d)
(province c)
)
(:goal
(and
(craves b c)
(craves c d)
(craves d a))
)
)


