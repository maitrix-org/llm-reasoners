

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b d)
(on c a)
(ontable d)
(clear b)
(clear c)
)
(:goal
(and
(on a c)
(on d a))
)
)


