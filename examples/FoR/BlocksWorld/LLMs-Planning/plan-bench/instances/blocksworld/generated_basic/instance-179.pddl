

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(on b c)
(on c a)
(ontable d)
(clear b)
)
(:goal
(and
(on a b)
(on c a))
)
)


