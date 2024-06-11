

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(on a b)
(on b c)
(ontable c)
(clear a)
)
(:goal
(and
(on a c)
(on b a))
)
)


