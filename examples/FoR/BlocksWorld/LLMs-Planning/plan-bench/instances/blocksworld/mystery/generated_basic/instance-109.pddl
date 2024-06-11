

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(planet b)
(craves c a)
(planet d)
(province b)
(province c)
)
(:goal
(and
(craves b a)
(craves d b))
)
)


